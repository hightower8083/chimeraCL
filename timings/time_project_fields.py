from time import time
import sys

from methods.generic_methods_cl import Communicator
from particles import Particles
from grid import Grid

def run_test(dims=(1024,256),Np=3e6,answer=2,verb=False,
             aligned=False, Nint = 100, Nheatup = 10):

    comm = Communicator(answers=[0,answer])
    grid_in = {'Xmin':-1.,'Xmax':1.,'Nx':dims[0],
               'Rmin':0,'Rmax':1.,'Nr':dims[1],
               'M':0}

    parts = Particles(grid_in,comm)
    grid = Grid(grid_in,comm)

    beam_in = {'Np':int(Np),
               'x_c':0.,'Lx':0.2,
               'y_c':0.,'Ly':0.2,
               'z_c':0.,'Lz':0.2,
               'px_c':0.,'dpx':0.5,
               'py_c':0.,'dpy':0.5,
               'pz_c':0.,'dpz':0.5}

    parts.make_parts(beam_in)
    parts.sort_parts()
    if aligned:
        parts.align_parts()
    grid.depose_charge([parts,])

    for i in range(Nint+Nheatup):
        if i==Nheatup: t0 = time()
        grid.project_fields([parts,])


    comm.thr.synchronize()
    timing_avrg = (time()-t0)/Nint*1e3
    if verb:
        print( "Timing averaged over {:d} loops is {:g} ms".
               format(Nint,timing_avrg) )

    return timing_avrg

if __name__ == "__main__":
    run_test(answer=int(sys.argv[-1]),verb=True)
