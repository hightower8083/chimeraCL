import numpy as np
from time import time
import sys

from chimeraCL.methods.generic_methods_cl import Communicator
from chimeraCL.particles import Particles
#from chimeraCL.grid import Grid
from chimeraCL.solver import Solver

def run_test(dims=(1024,256),Np=2e6,answers=[],verb=False,
             aligned=False, Nint = 100, Nheatup = 10):
    comm = Communicator(answers=answers)
    grid_in = {
        'Xmin':-1.,'Xmax':1.,'Nx':1024,
        'Rmin':0,'Rmax':1.,'Nr':200,
        'M':1}

    parts = Particles(grid_in,comm)
    grid = Solver(grid_in,comm)

    beam_in = {'Np':int(7*10**6),
               'x_c':0.,'Lx':0.3,
               'y_c':0.2,'Ly':0.3,
               'z_c':0.2,'Lz':0.3,
               'px_c':0.,'dpx':0.,
               'py_c':0.,'dpy':0.,
               'pz_c':0.,'dpz':0.,
              }

    parts.make_parts(beam_in)
    parts.sort_parts(grid=grid)
    parts.align_parts()
    grid.depose_charge([parts,])

    tmp0 = grid.DataDev['rho_m0'].get().copy()
    tmp1 = grid.DataDev['rho_m1'].get().copy()

    grid.fb_transform(comps = ['rho',],dir=0)
    grid.set_to_zero(grid.DataDev['rho_m0'])
    grid.set_to_zero(grid.DataDev['rho_m1'])
    grid.fb_transform(comps = ['rho',],dir=1)

    err_xr = (np.abs(grid.DataDev['rho_m0'].get()-tmp0)[1:] /
              np.abs(tmp0[1:]).max() +
              np.abs(grid.DataDev['rho_m1'].get()-tmp1)[1:] /
              np.abs(tmp1[1:]).max()).max()

    comm.thr.synchronize()
    if verb:
        print( "Error in transform is {:g}".
               format(err_xr) )
    return err_xr

if __name__ == "__main__":
    from numpy import array,int32
    conv_to_list = lambda str_var: list(array( str_var.split(':')).\
                                          astype(int32))

    if len(sys.argv)>1:
        run_test(answers=conv_to_list(sys.argv[-1]),verb=True)
    else:
        run_test(verb=True)
