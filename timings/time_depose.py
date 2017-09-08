from time import time
import sys

from methods.generic_methods_cl import Communicator
from particles import Particles
from grid import Grid

comm = Communicator(answers=[0,int(sys.argv[-1])])
grid_in = {'Xmin':-1.,'Xmax':1.,'Nx':1024,
           'Rmin':0,'Rmax':1.,'Nr':256,
           'M':0}

parts = Particles(grid_in,comm)
grid = Grid(grid_in,comm)

beam_in = {'Np':int(2e6),
           'x_c':0.,'Lx':0.2,
           'y_c':0.,'Ly':0.2,
           'z_c':0.,'Lz':0.2,
           'px_c':0.,'dpx':0.5,
           'py_c':0.,'dpy':0.5,
           'pz_c':0.,'dpz':0.5}

parts.make_parts(beam_in)

Nint = 100
Nheatup = 10
parts.sort_parts()
#parts.align_parts()

for i in range(Nint+Nheatup):
    if i==Nheatup: t0 = time()
    grid.depose_charge([parts,])

comm.thr.synchronize()
print( "Timing averaged over {:d} loops is {:g} ms".
      format(Nint,(time()-t0)/Nint*1e3) )

