import numpy as np
from time import time
import sys

from chimeraCL.methods.generic_methods_cl import Communicator
from chimeraCL.particles import Particles
from chimeraCL.grid import Grid

def run_test(dims=(1024,256),Np=2e6,answers=[0,2],verb=False,
             aligned=False, Nint = 100, Nheatup = 10):
    comm = Communicator(answers=answers)
    grid_in = {
        'Xmin':-1.,'Xmax':1.,'Nx':1024,
        'Rmin':0,'Rmax':1.,'Nr':200,
        'M':0}

    parts = Particles(grid_in,comm)
    grid = Grid(grid_in,comm)

    beam_in = {'Np':int(7*10**6),
               'x_c':0.,'Lx':0.3,
               'y_c':0.,'Ly':0.3,
               'z_c':0.,'Lz':0.3,
               'px_c':0.,'dpx':0.,
               'py_c':1.,'dpy':0.,
               'pz_c':0.,'dpz':0.,
              }

    parts.make_parts(beam_in)
    parts.sort_parts(grid=grid)
    parts.align_parts()


    xx = parts.DataDev['x'].get()
    rr = np.sqrt(parts.DataDev['y'].get()**2 + parts.DataDev['z'].get()**2)

    dat = grid.Args['Xgrid'][None,:]*grid.Args['Rgrid'][:,None]
    grid.DataDev['Ex_m0'][:] = dat.astype(grid.DataDev['Ex_m0'].dtype)

    grid.project_fields([parts,])
    comm.thr.synchronize()
    err_xr = np.abs(parts.DataDev['Ex'].get()-(xx*rr)).max()
    comm.thr.synchronize()
    if verb:
        print( "Error in projection of mode {:d} is {:g}".
               format(grid_in['M'], err_xr) )
    return err_xr

if __name__ == "__main__":
    from numpy import array,int32
    conv_to_list = lambda str_var: list(array( str_var.split(':')).\
                                          astype(int32))

    run_test(answers=conv_to_list(sys.argv[-1]),verb=True)
