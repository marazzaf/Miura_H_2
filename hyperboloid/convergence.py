#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from firedrake.petsc import PETSc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import sys

# the coefficient functions
def p(phi):
  sq = inner(phi.dx(0), phi.dx(0))
  aux = 4 / (4 - sq )
  return conditional(gt(sq, Constant(4)), Constant(100), aux)
  #return interpolate(conditional(lt(aux, Constant(1)), Constant(100), aux), UU)
  #return 1 / (1 - 0.25 * inner(phi.dx(0), phi.dx(0)))
  
def q(phi):
  sq = inner(phi.dx(1), phi.dx(1))
  aux = 4 / sq
  return conditional(gt(sq, Constant(4)), Constant(1), aux)

# Size for the domain
theta = pi/2
L = 2*sin(0.5*acos(0.5/cos(0.5*theta))) #length of rectangle
alpha = sqrt(1 / (1 - sin(theta/2)**2))
H = 2*pi/alpha #height of rectangle
l = sin(theta/2)*L

#Creating mesh
size_ref = 50 #10 #degub: 5
#mesh = RectangleMesh(size_ref, size_ref, L, H)
mesh = Mesh('mesh/convergence_2.msh')
V = VectorFunctionSpace(mesh, "BELL", 5, dim=3)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#For projection
UU = FunctionSpace(mesh, 'CG', 4)

#  Dirichlet boundary conditions
x = SpatialCoordinate(mesh)
rho = sqrt(4*cos(theta/2)**2*(x[0]-L/2)**2 + 1)
z = 2*sin(theta/2) * (x[0]-L/2)
phi_D = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), z))

#initial guess
W = VectorFunctionSpace(mesh, 'CG', 4, dim=3)
phi_l = Function(W, name='solution')
phi_t = TrialFunction(W)
psi = TestFunction(W)
laplace = inner(grad(phi_t), grad(psi)) * dx #laplace in weak form
L = inner(Constant((0,0,0)), psi) * dx
bcs = [DirichletBC(W, phi_D, 1), DirichletBC(W, phi_D, 2), DirichletBC(W, phi_D, 3), DirichletBC(W, phi_D, 4)]
#solve laplace equation on the domain
A = assemble(laplace, bcs=bcs)
b = assemble(L, bcs=bcs)
solve(A, phi_l, b, solver_parameters={'direct_solver': 'mumps'})
PETSc.Sys.Print('Laplace equation ok')

#Writing our problem now
phi = Function(V, name='solution')
phi.vector()[:] = project(phi_l, V).vector()
phi_t = TrialFunction(V)
psi = TestFunction(V)
#bilinear form for linearization
Gamma = (p(phi) + q(phi)) / (p(phi)*p(phi) + q(phi)*q(phi))
a = Gamma * inner(p(phi) * phi_t.dx(0).dx(0) + q(phi)*phi_t.dx(1).dx(1), div(grad(psi))) * dx
#a = inner(p(phi) * phi_t.dx(0).dx(0) + q(phi)*phi_t.dx(1).dx(1), div(grad(psi))) * dx

#penalty to impose Dirichlet BC
h = CellDiameter(mesh)
pen = 1e1 #1e1
pen_term = pen/h**4 * inner(phi_t, psi) * ds
a += pen_term
L = pen/h**4 * inner(phi_D, psi)  * ds

# Picard iteration
tol = 1e-5 #1e-9
maxiter = 50
phi_old = Function(V) #for iterations
for iter in range(maxiter):
  #linear solve
  A = assemble(a)
  b = assemble(L)
  pp = interpolate(p(phi), UU)
  PETSc.Sys.Print('Min of p: %.3e' % pp.vector().array().min())
  solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'}) # compute next Picard iterate
    
  eps = sqrt(assemble(inner(div(grad(phi-phi_old)), div(grad(phi-phi_old)))*dx)) # check increment size as convergence test
  PETSc.Sys.Print('iteration{:3d}  H2 seminorm of delta: {:10.2e}'.format(iter+1, eps))

  if eps < tol:
    break
  phi_old.assign(phi)

if eps > tol:
  PETSc.Sys.Print('no convergence after {} Picard iterations'.format(iter+1))
else:
  PETSc.Sys.Print('convergence after {} Picard iterations'.format(iter+1))

#Computing error
err = sqrt(assemble(inner(div(grad(phi-phi_D)), div(grad(phi-phi_D)))*dx))
PETSc.Sys.Print('Error: %.3e' % err)

#For projection
U = VectorFunctionSpace(mesh, 'CG', 4, dim=3)
#projected = project(phi, U, name='surface')

#Write 3d results
file = File('hyper.pvd')
x = SpatialCoordinate(mesh)
projected = Function(U, name='surface')
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file.write(projected)

#Write 2d result
file_bis = File('flat.pvd')
proj = project(phi, U, name='flat')
file_bis.write(proj)

