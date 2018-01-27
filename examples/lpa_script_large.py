"""

Example is inspired by the simulation presented in
R. Lehe et al, Computer Physics Communications 203, 66 (2016)

"""

import sys
from time import time
from copy import deepcopy
import numpy as np

from chimeraCL.methods.generic_methods_cl import Communicator
from chimeraCL.particles import Particles
from chimeraCL.solver import Solver
from chimeraCL.frame import Frame
from chimeraCL.laser import add_gausian_pulse
from chimeraCL.diagnostics import Diagnostics
from chimeraCL.pic_loop import PIC_loop

########################################
############ USER-END SETUP ############
### NB: mind numbers dtypes ############
########################################

# Simulation steps
Nsteps = 20000

# Diagnostics
diag_in = {'Interval': 1000,
           'ScalarFields': ['rho', 'Ex', 'Ez'],
           'Species':{'Components': ['x', 'y', 'z', 'w', 'px'],
                       'Selections': [['px', 5, None], ]} }

# Grid
xmin, xmax = -100., 40.
rmin, rmax = 0., 50.
Nx, Nr, M = 4*1024, 252, 1

# Laser
lam0 = 0.8 # laser wavelength

a0 = 4./lam0
Lx, w0 = 10., 16.
x0, x_foc = 0., 0.

# Plasma
dens = 0.5e18 / 1.1e21
Npx, Npr, Npth = 2, 2, 4

# Frame (maganes plasma injection at right boundary)
frame_velocity = 1.
frameSteps = 20
dens_profiles = [{'coord':'x',
                  'points':[-200, 40, 140, 340, 440, 1000],
                  'values':[   0,  0,   2,   2,   1,    1]}, ]

####################################################################
### SIMULATION CONSTRUCTOR (don't touch without asking me first) ###
####################################################################

comm = Communicator()

grid_in = {'Xmin': xmin, 'Xmax': xmax, 'Nx': Nx,
           'Rmin': rmin, 'Rmax': rmax, 'Nr': Nr,
           'M': M, 'DampCells': 50
          }

laser_in = {
    'k0': 1., 'a0': a0, 'x0': x0,
    'Lx': Lx, 'R': w0, 'x_foc': x_foc}

grid_in['dt'] = (grid_in['Xmax']-grid_in['Xmin']) / grid_in['Nx']

solver = Solver(grid_in, comm)
add_gausian_pulse(solver, laser=laser_in)

eons_in = {'Nppc': (Npx, Npr, Npth),
           'dx': solver.Args['dx'],
           'dr': solver.Args['dr'],
           'dt': solver.Args['dt'],
           'dens': dens,
           'charge': -1,
          }

ions_in = deepcopy(eons_in)
ions_in['charge'] = 1
ions_in['Immobile'] = True

eons = Particles(eons_in, comm)
ions = Particles(ions_in, comm)
ions.Args['InjectorSource'] = eons

frame_in = {'Velocity': frame_velocity,
            'dt': solver.Args['dt'],
            'Steps': frameSteps,
            'DensityProfiles': dens_profiles
           }

frame = Frame(frame_in)

diag = Diagnostics(solver=solver, species=[eons, ],
                   frame=frame, configs_in = diag_in)

loop = PIC_loop(solvers=[solver, ], species=[eons, ions],
                frames=[frame, ], diags = [diag, ])

############################################
############ RUN THE SIMULATION ############
############################################

t0 = time()
while loop.it<Nsteps+1:
    loop.step()
    if np.mod(loop.it, 10) == 0:
        sys.stdout.write("\rstep {:d} of {:d}".format(loop.it, Nsteps))
        sys.stdout.flush()

comm.queue.finish()
t0 = time() - t0
print("\nTotal time is {:g} mins \nMean step time is {:g} ms ".\
      format(t0/60., t0/Nsteps*1e3) )
