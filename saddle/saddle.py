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

# Create mesh and define function space
L = 2 #length of rectangle
H = 1 #height of rectangle #1.2 works #1.3 no
size_ref = 50 #degub: 2
mesh = RectangleMesh(size_ref, size_ref, L, H, diagonal='crossed')
#mesh = UnitDiskMesh(size_ref)
V = VectorFunctionSpace(mesh, "BELL", 5, dim=3)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#For projection
U = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
UU = FunctionSpace(mesh, 'CG', 4)
P = Function(UU)

# Boundary conditions
beta = 1 #0.1
x = SpatialCoordinate(mesh)
phi_D1 = beta*as_vector((x[0], x[1], 0))

#modify this one to be the right BC
alpha = pi/2 #pi/4
#modify the rest of the BC because it does not give the expected result...
l = H*L / sqrt(L*L + H*H)
sin_gamma = H / sqrt(L*L+H*H)
cos_gamma = L / sqrt(L*L+H*H)
DB = l*as_vector((sin_gamma,cos_gamma,0))
DBp = l*sin(alpha)*Constant((0,0,1)) + cos(alpha) * DB
OC = as_vector((L, 0, 0))
CD = Constant((-sin_gamma*l,H-cos_gamma*l,0))
OBp = OC + CD + DBp
BpC = -DBp - CD
BpA = BpC + Constant((-L, H, 0))
phi_D2 = (1-x[0]/L)*BpA + (1-x[1]/H)*BpC + OBp
phi_D2 *= beta

# Creating function to store solution
phi = Function(V, name='solution')
phi_old = Function(V) #for iterations

#Defining the bilinear forms
#bilinear form for linearization
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
#L = pen/h**4 * inner(phi_D1, psi) * ds
L = pen/h**4 * inner(phi_D1, psi) *(ds(1)+ds(3)) + pen/h**4 * inner(phi_D2, psi) *(ds(2)+ds(4))

#Computing initial guess
laplace = inner(grad(phi_t), grad(psi)) * dx #laplace in weak form
#laplace = inner(div(grad(phi_t)), div(grad(psi))) * dx #test
A = assemble(laplace+pen_term)
b = assemble(L)
solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'})
PETSc.Sys.Print('Laplace equation ok')

#test
Norm = sqrt(inner(grad(phi), grad(phi)))
file_bis = File('grad.pvd')
proj = project(Norm, UU, name='norm grad')
file_bis.write(proj)

#test
eps = sqrt(assemble(inner(div(grad(phi-phi_old)), div(grad(phi-phi_old)))*dx)) # check increment size as convergence test
PETSc.Sys.Print('Before computation  H2 seminorm of delta: {:10.2e}'.format(eps))

# Picard iterations
tol = 1e-5 #1e-9
maxiter = 100
for iter in range(maxiter):
  #linear solve
  A = assemble(a)
  b = assemble(L)
  solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'}) # compute next Picard iterate
  
  #convergence test 
  eps = sqrt(assemble(inner(div(grad(phi-phi_old)), div(grad(phi-phi_old)))*dx)) # check increment size as convergence test
  PETSc.Sys.Print('iteration{:3d}  H2 seminorm of delta: {:10.2e}'.format(iter+1, eps))
  if eps < tol:
    break
  phi_old.assign(phi)

if eps > tol:
  PETSc.Sys.Print('no convergence after {} Picard iterations'.format(iter+1))
else:
  PETSc.Sys.Print('convergence after {} Picard iterations'.format(iter+1))

#Write 2d results
flat = File('flat_%i.pvd' % size_ref)
W = VectorFunctionSpace(mesh, 'CG', 4, dim=3)
proj = project(phi, W, name='flat')
flat.write(proj)
  
#Write 3d results
file = File('new_%i.pvd' % size_ref)
x = SpatialCoordinate(mesh)
projected = Function(W, name='surface')
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file.write(projected)

#Test is inequalities are true
file_bis = File('verif_x.pvd')
phi_x = interpolate(phi.dx(0), W)
proj = project(inner(phi_x,phi_x), UU, name='test phi_x')
file_bis.write(proj)
file_ter = File('verif_y.pvd')
phi_y = interpolate(phi.dx(1), W)
proj = project(inner(phi_y,phi_y), UU, name='test phi_y')
file_ter.write(proj)
file_4 = File('verif_prod.pvd')
proj = project(inner(phi_x,phi_y), UU, name='test PS')
file_4.write(proj)

#Test
test = project(div(grad(phi)), W, name='minimal')
file_6 = File('minimal_bis.pvd')
file_6.write(test)

#Write 2d result for the grad
Norm = sqrt(inner(grad(phi), grad(phi)))
file_bis = File('grad.pvd')
proj = project(Norm, UU, name='norm grad')
file_bis.write(proj)
