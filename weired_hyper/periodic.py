#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from firedrake.petsc import PETSc
import sys

# the coefficient functions
def p(phi):
  aux = 1 / (1 - 0.25 * inner(phi.dx(0), phi.dx(0)))
  return interpolate(conditional(lt(aux, Constant(1)), Constant(100), aux), UU)

def q(phi):
  return 4 / inner(phi.dx(1), phi.dx(1))

def sq_norm(f):
  return inner(f, f)

#geometric parameters
theta = pi/2
L = 2*sin(0.5*acos(0.5/cos(0.5*theta))) #length of rectangle
alpha = sqrt(1 / (1 - sin(theta/2)**2))
H = 2*pi/alpha #height of rectangle

# Create mesh and define function space
size_ref = 20 #degub: 5
mesh = PeriodicRectangleMesh(size_ref, size_ref, L, H, direction='y', diagonal='crossed')
V = VectorFunctionSpace(mesh, "BELL", 5, dim=3)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#For projection
U = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
UU = FunctionSpace(mesh, 'CG', 4)
W = VectorFunctionSpace(mesh, 'CG', 4, dim=3)

# Boundary conditions
x = SpatialCoordinate(mesh)
rho = sqrt(4*cos(theta/2)**2*(L/2)**2 + 1)
#BC on lower part
phi_D1 = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), -2*sin(theta/2)*L))
#BC on upper part
beta = pi/4
phi_D2 = as_vector((rho*cos(beta)*cos(alpha*x[1]), rho*cos(beta)*sin(alpha*x[1]), rho*sin(beta)*cos(alpha*x[1])))
# Creating function to store solution
phi = Function(V, name='solution')
phi_old = Function(V) #for iterations

#Defining the bilinear forms for linearization
phi_t = TrialFunction(V)
psi = TestFunction(V)
a = inner(p(phi) * phi_t.dx(0).dx(0) + q(phi)*phi_t.dx(1).dx(1), div(grad(psi))) * dx

#penalty to impose Dirichlet BC
h = CellDiameter(mesh)
pen = 1e1
#lhs
pen_term = pen/h**4 * inner(phi_t, psi) * ds
a += pen_term
#rhs
LL = pen/h**4 * inner(phi_D1, psi) * ds(1) + pen/h**4 * inner(phi_D2, psi) * ds(2)

#Computing initial guess
laplace = inner(grad(phi_t), grad(psi)) * dx #laplace in weak form
A = assemble(laplace+pen_term)
b = assemble(LL)
solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'})
PETSc.Sys.Print('Laplace equation ok')

#Newton bilinear form
Gamma = (p(phi) + q(phi)) / (p(phi)*p(phi) + q(phi)*q(phi))
a = Gamma * inner(p(phi) * phi.dx(0).dx(0) + q(phi)*phi.dx(1).dx(1), div(grad(psi))) * dx

#penalty to impose Dirichlet BC
pen_term = pen/h**4 * inner(phi, psi) * ds
a += pen_term
a -= LL

# Solving with Newton method
solve(a == 0, phi, solver_parameters={'snes_monitor': None, 'snes_max_it': 25})

#Write 2d results
flat = File('flat_%i.pvd' % size_ref)
#proj = project(phi, W, name='flat')
proj = Function(W, name='flat')
proj.interpolate(phi - 1e-10*as_vector((x[0], x[1], 0)))
flat.write(proj)
  
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

#test
Norm = inner(phi.dx(0), phi.dx(0))
file_bis = File('dx.pvd')
proj = project(Norm, UU, name='norm dx')
file_bis.write(proj)

sys.exit()

#Write 3d results
mesh = RectangleMesh(size_ref, size_ref, L*0.999, H*0.999, diagonal='crossed')
W = VectorFunctionSpace(mesh, 'CG', 4, dim=3)
X = interpolate(mesh.coordinates, VectorFunctionSpace(mesh, 'CG', 4))

#gives values from phi
def func(data):
  res = np.zeros((len(data),3))
  for i,dat in enumerate(data):
    res[i,:] = proj(dat)
  return res

# Use the external data function to interpolate the values of f.
phi_bis = Function(W)
phi_bis.dat.data[:] = func(X.dat.data_ro)

#interpolation on new mesh
projected = Function(W, name='surface')
file = File('new_%i.pvd' % size_ref)
x = SpatialCoordinate(mesh)
#projected.interpolate(as_vector((x[0], x[1], 0)))
projected.interpolate(phi_bis - as_vector((x[0], x[1], 0)))
file.write(projected)
