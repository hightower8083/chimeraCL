from time import time
import sys
from numpy import array, int32

from methods.generic_methods_cl import Communicator
from particles import Particles
from grid import Grid

def run_test(dims=(1024,256),Np=3e6,answers=[0,2],verb=False,
             aligned=False, Nint = 100, Nheatup = 10):

    if answers is None:
        return 0

    comm = Communicator(answers=answers)
    grid_in = {'Xmin':-1.,'Xmax':1.,'Nx':dims[0],
               'Rmin':0,'Rmax':1.,'Nr':dims[1],
               'M':1}

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
    parts.sort_parts(grid)
    if aligned:
        parts.align_parts()

    for i in range(Nint+Nheatup):
        if i==Nheatup: t0 = time()
        grid.depose_charge([parts,])

    comm.thr.synchronize()
    timing_avrg = (time()-t0)/Nint*1e3
    if verb:
        print( "Timing averaged over {:d} loops is {:g} ms".
               format(Nint,timing_avrg) )

    del parts
    del grid
    del comm
    return timing_avrg

if __name__ == "__main__":
    from numpy import array, int32
    conv_to_list = lambda str_var: list(array( str_var.split(':')).\
                                          astype(int32))

    run_test(answers=conv_to_list(sys.argv[-1]),verb=True)
