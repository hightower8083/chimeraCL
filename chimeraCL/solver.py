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

        self._make_ms_coefficients()
        self.send_args_to_dev()

    def push_fields(self):
        self.advance_fields(vecs=['E', 'G', 'J', 'dN0', 'dN1'])

    def damp_fields(self):
        self.fb_transform(vects=['E', 'G'], dir=1, mode='half')
        self.profile_edges(['E', 'G'])
        self.fb_transform(vects=['E', 'G'], dir=0, mode='half')

    def restore_B_fb(self):
        self.field_rot('G', 'B')
        self.field_poiss_vec('B')

    def _make_ms_coefficients(self):
        for m in range(self.Args['M']+1):
            w = self.Args['w_m'+str(m)]
            dt = self.Args['dt']

            self.Args['MxSlv_cos(wdt)_m'+str(m)] = np.cos(w * dt)
            self.Args['MxSlv_sin(wdt)*w_m'+str(m)] = np.sin(w*dt) * w
            self.Args['MxSlv_1/w**2_m'+str(m)] = 1. / w**2

            self.Args['dont_keep'].append('MxSlv_cos(wdt)_m'+str(m))
            self.Args['dont_keep'].append('MxSlv_sin(wdt)*w_m'+str(m))
            self.Args['dont_keep'].append('MxSlv_1/w**2_m'+str(m))

    def _make_ms_coefficients_old(self):
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
