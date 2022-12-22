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

# Size for the domain
theta = pi/2 #pi/2
L = 2*sin(0.5*acos(0.5/cos(0.5*theta))) #length of rectangle
alpha = sqrt(1 / (1 - sin(theta/2)**2))
H = 2*pi/alpha #height of rectangle
l = sin(theta/2)*L

#Creating mesh
size_ref = 10 #10 #degub: 5
mesh = PeriodicRectangleMesh(size_ref, size_ref, L, H, direction='y', diagonal='crossed')
V = VectorFunctionSpace(mesh, "BELL", 5, dim=3) #faster
VV = FunctionSpace(mesh, 'CG', 4)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#For projection
UU = FunctionSpace(mesh, 'CG', 4)

#  Dirichlet boundary conditions
x = SpatialCoordinate(mesh)
rho = sqrt(4*cos(theta/2)**2*(x[0]-L/2)**2 + 1)
z = 2*sin(theta/2) * (x[0]-L/2)
phi_D = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), z))

#initial guess
#solve laplace equation on the domain
phi = Function(V, name='solution')
phi_t = TrialFunction(V)
psi = TestFunction(V)
dx = dx(degree=5)
laplace = inner(grad(phi_t), grad(psi)) * dx #laplace in weak form
#penalty term for Dirichlet BC
h = CellDiameter(mesh)
pen = 1e1 #1e1
pen_term = pen/h**4 * inner(phi_t, psi) * (ds(1) + ds(2))
L = pen/h**4 * inner(phi_D, psi)  * (ds(1) + ds(2))
A = assemble(laplace+pen_term)
b = assemble(L)
solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'})
#solve(A, phi, b, solver_parameters={'ksp_type': 'cg','pc_type': 'bjacobi', 'ksp_rtol': 1e-5})
PETSc.Sys.Print('Laplace equation ok')


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
X = VectorFunctionSpace(mesh, 'CG', 2, dim=3)
projected = interpolate(div(grad(phi)), X)
ref = interpolate(div(grad(phi_D)), X)
err = sqrt(assemble(inner(div(grad(phi-phi_D)), div(grad(phi-phi_D)))*dx))
#PETSc.Sys.Print('Error: %.3e' % err)

#For projection
U = VectorFunctionSpace(mesh, 'CG', 4, dim=3)

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

