"""A Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.
This script demonstrates solving a 2D Cartesian initial value problem. It can
be ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_snapshots.py` script can be used to
produce plots from the saved data. It should take about 5 cpu-minutes to run.

The problem is non-dimensionalized using the box height and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)

For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 plot_snapshots.py snapshots/*.h5
"""

import argparse
import numpy as np
import dedalus.public as d3
from mpi4py import MPI
import logging
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD
ncpu = comm.size
log2 = np.log2(ncpu)
if log2 == int(log2):
    mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
logger.info("running on processor mesh={}".format(mesh))


#Parser
parser = argparse.ArgumentParser(
                    prog='rotating_rayleigh_benard.py',
                    description='A script for 3D simulations of rotating Rayleigh Benard convection. Defaults to low-Prandtl numbers to study oscillitory onset',
                    epilog='options below')
parser.add_argument('-a', '--aspect', default=4, type=float, help='Aspect ratio of simulation Lx and Ly are the same, aspect = Lx/Lz = Ly/Lz')
parser.add_argument('-R', '--Rayleigh', default=1e6, type=float, help='Rayleigh number')
parser.add_argument('-P', '--Prandtl', default=0.1, type=float, help='Prandtl number')
parser.add_argument('-T', '--Taylor', default=1e6, type=float, help='Taylor number')
parser.add_argument('--nhoriz', default=128, type=int, help='horizontal resolution')
parser.add_argument('--nz', default=32, type=int, help='vertical resolution')
parser.add_argument('--stop_sim_time', type=float, help='stop time in freefall timescales')
parser.add_argument('--stop_wall_time', type=float, help='wall stop time in hours')
parser.add_argument('--label', type=str, help='output label')
parser.add_argument('--restart', type=str, help='restarts simulation from specified file')
args=parser.parse_args()

# Parameters
Lx, Ly, Lz = args.aspect, args.aspect, 1
Nx, Ny, Nz = args.nhoriz, args.nhoriz, args.nz
Rayleigh = args.Rayleigh
Prandtl = args.Prandtl
Taylor = args.Taylor
dealias = 3/2
stop_sim_time = args.stop_sim_time
stop_wall_time = args.stop_wall_time
restart = args.restart
label= args.label
timestepper = d3.RK222
max_timestep = 0.125
dtype = np.float64



# Bases
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis, ybasis, zbasis))
b = dist.Field(name='b', bases=(xbasis, ybasis, zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis, ybasis, zbasis))
tau_p = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1', bases=(xbasis, ybasis))
tau_b2 = dist.Field(name='tau_b2', bases=(xbasis, ybasis))
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(xbasis, ybasis))
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(xbasis, ybasis))

# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
ex, ey, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
cross = lambda A,B: d3.cross(A,B)
curl = lambda A: d3.Curl(A)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction

Omega = 1/2*Taylor**(1/2) * (Rayleigh / Prandtl)**(-1/2) * ez

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"

trans = lambda A: d3.TransposeComponents(A)
e = d3.grad(u) + trans(d3.grad(u))
problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2 ) = - u@grad(b)")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + 2*cross(Omega, u) + lift(tau_u2) = - u@grad(u)")
problem.add_equation("b(z=0) = Lz")
#problem.add_equation("u(z=0) = 0")
problem.add_equation('ez@u(z=0) = 0')
problem.add_equation('ez@(ex@e(z=0)) = 0')
problem.add_equation('ez@(ey@e(z=0)) = 0')

problem.add_equation("b(z=Lz) = 0")
#problem.add_equation("u(z=Lz) = 0")
problem.add_equation('ez@u(z=Lz) = 0')
problem.add_equation('ez@(ex@e(z=Lz)) = 0')
problem.add_equation('ez@(ey@e(z=Lz)) = 0')
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
if stop_sim_time is not None:
    solver.stop_sim_time = stop_sim_time
if stop_wall_time is not None:
    solver.stop_wall_time = stop_wall_time*60*60
# Initial conditions
b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
b['g'] *= z * (Lz - z) # Damp noise at walls
b['g'] += Lz - z # Add linear background

# Analysis
outdir = 'rotating_RBC_Ra{:.2g}_Ta{:.2g}_Pr{:.2g}_nx{:}_ny{:}_nz{:}'.format(Rayleigh, Taylor, Prandtl, Nx, Ny, Nz)
if label is not None:
    outdir+='_'+label
snapshots = solver.evaluator.add_file_handler(outdir+'/snapshots', sim_dt=10, max_writes=10)
snapshots.add_task(b, name='buoyancy')
snapshots.add_task(u@ex, name='u')
snapshots.add_task(u@ey, name='v')
snapshots.add_task(u@ez, name='w')
#snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

slices = solver.evaluator.add_file_handler(outdir+'/slices', sim_dt=0.25, max_writes=50)
slices.add_task(b(z=0.1*Lz), name='b z=0.1')
slices.add_task(b(z=0.2*Lz), name='b z=0.2')
slices.add_task(b(z=0.5*Lz), name='b z=0.5')
slices.add_task(b(z=0.8*Lz), name='b z=0.8')
slices.add_task(b(z=0.9*Lz), name='b z=0.9')

slices.add_task(ex@u(z=0.1*Lz), name='u z=0.1')
slices.add_task(ex@u(z=0.2*Lz), name='u z=0.2')
slices.add_task(ex@u(z=0.5*Lz), name='u z=0.5')
slices.add_task(ex@u(z=0.8*Lz), name='u z=0.8')
slices.add_task(ex@u(z=0.9*Lz), name='u z=0.9')

slices.add_task(ey@u(z=0.1*Lz), name='v z=0.1')
slices.add_task(ey@u(z=0.2*Lz), name='v z=0.2')
slices.add_task(ey@u(z=0.5*Lz), name='v z=0.5')
slices.add_task(ey@u(z=0.8*Lz), name='v z=0.8')
slices.add_task(ey@u(z=0.9*Lz), name='v z=0.9')

slices.add_task(ez@u(z=0.1*Lz), name='w z=0.1')
slices.add_task(ez@u(z=0.2*Lz), name='w z=0.2')
slices.add_task(ez@u(z=0.5*Lz), name='w z=0.5')
slices.add_task(ez@u(z=0.8*Lz), name='w z=0.8')
slices.add_task(ez@u(z=0.9*Lz), name='w z=0.9')

slices.add_task(ez@curl(u)(z=0.1*Lz), name='z vorticity z=0.1')
slices.add_task(ez@curl(u)(z=0.2*Lz), name='z vorticity z=0.2')
slices.add_task(ez@curl(u)(z=0.5*Lz), name='z vorticity z=0.5')
slices.add_task(ez@curl(u)(z=0.8*Lz), name='z vorticity z=0.8')
slices.add_task(ez@curl(u)(z=0.9*Lz), name='z vorticity z=0.9')

checkpoints = solver.evaluator.add_file_handler(outdir+'checkpoints', wall_dt=4*60*60, max_writes=1)
checkpoints.add_tasks(solver.state)


vol = Lx*Ly*Lz
integ = lambda A: d3.Integrate(d3.Integrate(d3.Integrate(A, 'x'), 'y'), 'z')
avg = lambda A: integ(A)/vol

traces = solver.evaluator.add_file_handler(outdir+'/traces', sim_dt=0.05, max_writes=None)
traces.add_task(avg(np.sqrt(u@u)/nu), name='Re')
profiles = solver.evaluator.add_file_handler(outdir+'/profiles', sim_dt=0.25, max_writes=50)

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')

# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
