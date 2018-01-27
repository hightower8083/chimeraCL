import numpy as np
from time import time
import sys

from chimeraCL.methods.generic_methods_cl import Communicator
from chimeraCL.solver import Solver
from chimeraCL.laser import add_gausian_pulse

def run_test(answers=[],verb=False):

    comm = Communicator(answers=answers)
    grid_in = {}
    grid_in['Xmin'] = -50.
    grid_in['Xmax'] = 50.
    grid_in['Rmax'] = 50.

    dx, dr, grid_in['dt'] = 0.05, 0.3, 0.05
    grid_in['Nx'] = int((grid_in['Xmax']-grid_in['Xmin'])/dx)//2 * 2
    grid_in['Nr'] = int(grid_in['Rmax']/dr)//2 * 2 + 1
    grid_in['M'] = 0


    laser_in = {
        'k0':1., 'a0':1., 'x0':0,
        'Lx':12., 'R':12., 'x_foc':20.}

    solver = Solver(configs_in=grid_in, comm=comm)
    add_gausian_pulse(solver, laser=laser_in)

    xcentr1 = []
    xcentr2 = []

    for i in range(100):
        solver.push_fields()
        solver.fb_transform(scals=['Ez', ], dir=1)

        var0 = np.real(solver.DataDev['Ez_m0'].get())
        Px = (solver.Args['Rgrid'][1:,None]*var0[1:,:]**2).sum(0)
        xcentr1.append((solver.Args['Xgrid'][None,:]*Px).sum()/Px.sum())

        Px = var0[1,:]**2
        xcentr2.append((solver.Args['Xgrid'][None,:]*Px).sum()/Px.sum())

    xcentr = np.array(xcentr1)
    veloc_num = 1-(xcentr[1:]-xcentr[:-1])/solver.Args['dt']
    veloc_theory = (2.*np.pi*laser_in['R'])**-2
    err1 = np.abs(veloc_num-veloc_theory).mean()/veloc_theory

    xcentr = np.array(xcentr2)
    veloc_num = 1-(xcentr[1:]-xcentr[:-1])/solver.Args['dt']
    veloc_theory = 2*(2.*np.pi*laser_in['R'])**-2
    err2 = np.abs(veloc_num-veloc_theory).mean()/veloc_theory

    if verb:
        print("""Deviation from theory in laser 
    centroid velocity is {:g} %
    on-axis group velocity is {:g} %""".
               format(err1*100,err2*100,) )
    return err1+err2

if __name__ == "__main__":
    from numpy import array,int32
    conv_to_list = lambda str_var: list(array( str_var.split(':')).\
                                          astype(int32))

    if len(sys.argv)>1:
        run_test(answers=conv_to_list(sys.argv[-1]),verb=True)
    else:
        run_test(verb=True)
