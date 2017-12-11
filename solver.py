import numpy as np

from chimeraCL.transformer import Transformer
from chimeraCL.grid import Grid

from chimeraCL.methods.solver_methods_cl import SolverMethodsCL


class Solver(Grid, Transformer, SolverMethodsCL):
    def __init__(self, configs_in, comm):
        if comm is not None:
            self.import_comm(comm)

        self.set_global_working_group_size()

        self._process_configs(configs_in)
        self.Args['vec_comps'] = ['x','y','z']

        self.init_solver_methods()
        self.init_grid_methods()

        self.DataDev = {}
        self._init_grid_data_on_dev()
        self.init_transformer()
        self.send_args_to_dev()

        self._make_ms_coefficients()

    def push_fields(self):
        self.advance_fields(vecs=['E', 'G', 'J', 'dN0', 'dN1'])

    def add_gausian_pulse(self, laser):

        k0 = 2*np.pi*laser['k0']
        a0 = laser['a0']

        Lx = laser['Lx']
        R = laser['R']
        x0 = laser['x0']
        x_foc = laser['x_foc']
        X_focus = x0 - x_foc

        Xgrid,Rgrid = self.Args['Xgrid'],self.Args['Rgrid'] # sin phase
        kx,w = self.Args['kx'][None,:], self.Args['w_m0']

        self.DataDev['Ez_m0'][:] = a0 * np.sin(k0*(Xgrid[None,:]-x0)) \
              * np.exp(-(Xgrid[None,:]-x0)**2/Lx**2 - Rgrid[:,None]**2/R**2) \
              * (abs(Rgrid[:,None]) < 3.5*R) * (abs(Xgrid[None,:]-x0) < 3.5*Lx)

        self.fb_transform(comps=['Ez',],dir=0)
        EE = self.DataDev['Ez_fb_m0'].get()

        DT = -1.j*w*np.sign(kx + (kx==0))
        GG  = DT*EE

        EE_tmp = np.cos(w*X_focus)*EE + np.sin(w*X_focus)/w*GG
        GG = -w*np.sin(w*X_focus)*EE + np.cos(w*X_focus)*GG
        EE = EE_tmp
        EE *= np.exp(1.j * kx * X_focus)
        GG *= np.exp(1.j * kx * X_focus)

        self.DataDev['Ez_fb_m0'][:] = EE
        self.DataDev['Gz_fb_m0'][:] = GG

    def _make_ms_coefficients(self):
        for m in range(self.Args['M']+1):
            mstr = '_m'+str(m)
            w = self.Args['w_m'+str(m)]
            dt = self.Args['dt']
            self.DataDev['MxSlv_cos(wdt)'+mstr] = self.dev_arr(
              dtype=np.double, shape=(self.Args['Nr']-1, self.Args['Nx']))
            self.DataDev['MxSlv_sin(wdt)*w'+mstr] = self.dev_arr(
              dtype=np.double, shape=(self.Args['Nr']-1, self.Args['Nx']))
            self.DataDev['MxSlv_1/w**2'+mstr] = self.dev_arr(
              dtype=np.double, shape=(self.Args['Nr']-1, self.Args['Nx']))

            self.DataDev['MxSlv_cos(wdt)'+mstr][:] = np.cos(w * dt)
            self.DataDev['MxSlv_sin(wdt)*w'+mstr][:] = np.sin(w*dt) * w
            self.DataDev['MxSlv_1/w**2'+mstr][:] = 1. / w**2
