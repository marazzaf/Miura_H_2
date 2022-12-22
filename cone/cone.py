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
  
def q(phi):
  sq = inner(phi.dx(1), phi.dx(1))
  aux = 4 / sq
  return conditional(lt(sq, Constant(1)), Constant(4), aux)


def sq_norm(f):
  return inner(f, f)

# Create mesh and define function space
alpha = 1
L = 1/alpha #length of rectangle
H = pi/alpha #height of rectangle #1.2*pi does not work
size_ref = 40 #60 #20 #10 #degub: 5
#nx,ny = int(size_ref*L/H),int(size_ref*H/L)
#mesh = PeriodicRectangleMesh(nx, ny, L, H, direction='y', diagonal='crossed')
mesh = RectangleMesh(size_ref, size_ref, L, H, diagonal='crossed')
#V = VectorFunctionSpace(mesh, "ARG", 5, dim=3)
V = VectorFunctionSpace(mesh, "BELL", 5, dim=3)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#For projection
UU = FunctionSpace(mesh, 'CG', 4)

# Boundary conditions
x = SpatialCoordinate(mesh)
rho = x[0]
z = x[0]
phi_D = as_vector((rho*cos(x[1]*alpha), rho*sin(x[1]*alpha), z))

# Creating function to store solution
phi = Function(V, name='solution')
phi_old = Function(V) #for iterations

#Defining the bilinear forms
#bilinear form for linearization
phi_t = TrialFunction(V)
psi = TestFunction(V)
dx = dx(degree=5)
a = inner(p(phi) * phi_t.dx(0).dx(0) + q(phi)*phi_t.dx(1).dx(1), div(grad(psi))) * dx

#penalty to impose Dirichlet BC
h = CellDiameter(mesh)
pen = 1e1 #1e2
#lhs
pen_term = pen/h**4 * inner(phi_t, psi) * ds #h**2
a += pen_term
#rhs
L = pen/h**4 * inner(phi_D, psi) * ds #h**2

#Computing initial guess
laplace = inner(grad(phi_t), grad(psi)) * dx #laplace in weak form
A = assemble(laplace+pen_term)
b = assemble(L)
solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'})
PETSc.Sys.Print('Laplace equation ok')

#testing bounded slope condition of initial guess
test = project(sq_norm(phi.dx(0)), UU)
with test.dat.vec_ro as v:
    value = v.max()[1]
#value = max(test.vector())
#sys.exit()
try:
  assert value < 4 #not elliptic otherwise.
except AssertionError:
  PETSc.Sys.Print('Bouned slope condition %.2e' % value)

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

  
## Picard iterations
#tol = 1e-5 #1e-9
#maxiter = 50
#for iter in range(maxiter):
#  #linear solve
#  A = assemble(a)
#  b = assemble(L)
#  solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'}) # compute next Picard iterate
#
#  #ellipticity test
#  test = project(sq_norm(phi.dx(0)), UU)
#  with test.dat.vec_ro as v:
#    value = v.max()[1]
#  try:
#    assert value < 4 #not elliptic otherwise.
#  except AssertionError:
#    PETSc.Sys.Print('Bouned slope condition %.2e' % value)
#    #Plot(test)
#    #sys.exit()
#  
#  #convergence test 
#  eps = sqrt(assemble(inner(div(grad(phi-phi_old)), div(grad(phi-phi_old)))*dx)) # check increment size as convergence test
#  PETSc.Sys.Print('iteration{:3d}  H2 seminorm of delta: {:10.2e}'.format(iter+1, eps))
#  if eps < tol:
#    break
#  phi_old.assign(phi)
#
#if eps > tol:
#  PETSc.Sys.Print('no convergence after {} Picard iterations'.format(iter+1))
#else:
#  PETSc.Sys.Print('convergence after {} Picard iterations'.format(iter+1))

#For projection
U = VectorFunctionSpace(mesh, 'CG', 4, dim=3)
  
#Write 3d results
file = File('cone.pvd')
x = SpatialCoordinate(mesh)
projected = Function(U, name='surface')
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file.write(projected)

#Write 2d result
file_bis = File('flat.pvd')
proj = project(phi, U, name='flat')
file_bis.write(proj)

#Test is inequalities are true
file_bis = File('verif_x.pvd')
phi_x = interpolate(phi.dx(0), U)
proj = project(inner(phi_x,phi_x), UU, name='test phi_x')
file_bis.write(proj)
file_ter = File('verif_y.pvd')
phi_y = interpolate(phi.dx(1), U)
proj = project(inner(phi_y,phi_y), UU, name='test phi_y')
file_ter.write(proj)
file_4 = File('verif_prod.pvd')
proj = project(inner(phi_x,phi_y), UU, name='test PS')
file_4.write(proj)
