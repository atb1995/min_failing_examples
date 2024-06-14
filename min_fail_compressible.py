from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate,
                       Constant, pi, cos, Function, sqrt, conditional, exp,
                       as_vector)
from gusto import thermodynamics as tde
import sys 
# ------------------------------------------------------------------------ #
# Set up model objects
# ------------------------------------------------------------------------ #

# Domain
dt = 6.0
tmax = 10*dt
nlayers = 120  # horizontal layers
ncols = 120  # number of columns
Lx = 1000.0
Lz = 1000.0
mesh_name = 'dry_compressible_mesh'
m = PeriodicIntervalMesh(ncols, Lx)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=Lz/nlayers, name=mesh_name)
domain = Domain(mesh, dt, "CG", 1)

# Equation
parameters = CompressibleParameters()
eqn = CompressibleEulerEquations(domain, parameters)

# I/O
output = OutputParameters(dirname="dry_compressible",
                          dumplist=['u','rho','theta'],
                          dump_vtus=True,
                          dump_nc=False,
                            dumpfreq=2)
io = IO(domain, output)

# Transport schemes
transported_fields = []
transport_methods = [DGUpwind(eqn, 'u'),
                        DGUpwind(eqn, 'rho'),
                        DGUpwind(eqn, 'theta')]

# Linear solver
linear_solver = CompressibleSolver(eqn)

# Time stepper
stepper = SemiImplicitQuasiNewton(eqn, io, transported_fields,
                                    transport_methods,
                                    linear_solver=linear_solver,
                                    num_outer=4, num_inner=1)

# ------------------------------------------------------------------------ #
# Initial conditions
# ------------------------------------------------------------------------ #

R_d = parameters.R_d
g = parameters.g

rho0 = stepper.fields("rho")
theta0 = stepper.fields("theta")
u0 = stepper.fields("u")

# Approximate hydrostatic balance
x, z = SpatialCoordinate(mesh)
T = Constant(300.0)
zH = R_d * T / g
p = Constant(100000.0) * exp(-z / zH)
theta0.interpolate(tde.theta(parameters, T, p))
rho0.interpolate(p / (R_d * T))

stepper.set_reference_profiles([('rho', rho0), ('theta', theta0)])

# # Add perturbation
r = sqrt((x-Lx/2)**2 + (z-Lz/2)**2)
theta_pert = 1.0*exp(-(r/(Lx/5))**2)
theta0.interpolate(theta0 + theta_pert)

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)