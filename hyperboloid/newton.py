#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from firedrake.petsc import PETSc
import sys

# the coefficient functions
def p(phi):
  sq = inner(phi.dx(0), phi.dx(0))
  aux = 4 / (4 - sq )
  return conditional(gt(sq, Constant(4)), Constant(100), aux)
  
def q(phi):
  sq = inner(phi.dx(1), phi.dx(1))
  aux = 4 / sq
  return conditional(lt(sq, Constant(1)), Constant(4), aux)

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

#test
Norm = sqrt(inner(grad(phi_l), grad(phi_l)))
file_bis = File('grad.pvd')
proj = project(Norm, UU, name='norm grad')
file_bis.write(proj)

#Writing our problem now
phi = Function(V, name='solution')
phi.vector()[:] = project(phi_l, V).vector()
psi = TestFunction(V)

#Newton bilinear form
Gamma = (p(phi) + q(phi)) / (p(phi)*p(phi) + q(phi)*q(phi))
a = Gamma * inner(p(phi) * phi.dx(0).dx(0) + q(phi)*phi.dx(1).dx(1), div(grad(psi))) * dx

#penalty to impose Dirichlet BC
h = CellDiameter(mesh)
pen = 1e1 #1e1
pen_term = pen/h**4 * inner(phi, psi) * ds
a += pen_term
L = pen/h**4 * inner(phi_D, psi)  * ds
a -= L

# Solving with Newton method
solve(a == 0, phi, solver_parameters={'snes_monitor': None, 'snes_max_it': 25})

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

#Write 2d result for the grad
Norm = sqrt(inner(grad(phi), grad(phi)))
file_bis = File('grad_2.pvd')
proj = project(Norm, UU, name='norm grad')
file_bis.write(proj)

