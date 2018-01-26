import numpy as np
from time import time
import sys
from copy import deepcopy

from chimeraCL.methods.generic_methods_cl import Communicator
from chimeraCL.particles import Particles
from chimeraCL.solver import Solver
from chimeraCL.frame import Frame
from chimeraCL.laser import add_gausian_pulse

from chimeraCL.diagnostics import Diagnostics
from chimeraCL.pic_loop import PIC_loop

comm = Communicator(answers=[0,])

# Simulation steps
Nsteps = 20000

# Grid
xmin, xmax = -100/0.8, 40/0.8
rmin, rmax = 0., 50./0.8
Nx, Nr, M = 4*1024, 252, 1

# Laser
a0 = 4.
Lx, w0 = 10./0.8, 16./0.8
x0, x_foc = 0., 0.

# Plasma
dens = 0.5e18 / (1.1e21/0.8**2)
Npx, Npr, Npth = 2, 2, 4

# Frame
velocity = 1.
frameSteps = 20
dens_profile = {'coord':'x',
                'points':[-200, 40/0.8, 140/0.8, 340/0.8, 440/0.8, 1000/0.8],
                'values':[   0,      0,       2,       2,       1,        1]
                }
# Diagnostics
diag = Diagnostics(
    solver=solver, species=[eons, ],
    configs_in={'Interval':100,
                'ScalarFields':['rho', 'Ez', 'Ex'],
                'Species':{'Components':['x', 'y', 'z', 'w', 'px'],
                           'Selections':[['px', 5, None],]}
               }  )


# Simulation objects setup

grid_in = {'Xmin':xmin, 'Xmax':xmax, 'Nx':Nx,
           'Rmin':rmin, 'Rmax':rmax, 'Nr':Nr,
           'M':M, 'DampCells':50
          }

grid_in['dt'] = (grid_in['Xmax']-grid_in['Xmin'])/grid_in['Nx']

laser_in = {
    'k0':1., 'a0':a0, 'x0':x0,
    'Lx':lx, 'R':w0, 'x_foc':x_foc}

solver = Solver(grid_in,comm)
add_gausian_pulse(solver, laser=laser_in)

eons_in = {'Nppc':(Npx, Npr, Npth),
           'dx':solver.Args['dx'],
           'dr':solver.Args['dr'],
           'dt':solver.Args['dt'],
           'dens': dens,
           'charge':-1,
          }

ions_in = deepcopy(eons_in)
ions_in['charge'] = 1
ions_in['Immobile'] = True

eons = Particles(eons_in,comm)
ions = Particles(ions_in,comm)

ions.Args['InjectorSource'] = eons

frame_in = {'Velocity':velocity,
            'dt':solver.Args['dt'],
            'Steps':frameSteps,
            'DensityProfiles':[dens_profile,]
           }

frame = Frame(frame_in)

loop = PIC_loop(solvers=[solver, ], species=[eons, ions],
                frames=[frame, ], diags = [diag, ])

# Running the simulation
t0 = time()
while loop.it<Nsteps+1:
    loop.step()
    if np.mod(loop.it, 10) == 0:
        sys.stdout.write("\rstep {:s} of {:s}".format(loop.it, Nsteps))
        sys.stdout.flush()

comm.queue.finish()
t0 = time() - t0
print("\nTotal time is {:g} mins \nMean step time is {:g} ms ".\
      format(t0/60., t0/Nsteps*1e3) )
