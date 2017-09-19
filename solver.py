import numpy as np
from chimeraCL.transformer import Transformer
from chimeraCL.grid import Grid

class Solver(Grid, Transformer):
    def __init__(self, configs_in, comm):
        if comm is not None:
            self.import_comm(comm)
        self._process_configs(configs_in)
        self.init_grid_methods()
        self.init_transformer()

        self._make_ms_coefficients()

        self.send_args_to_dev()
        self._init_data_on_dev()

    def advance_fields(self):
        pass

    def _make_ms_coefficients(self):
        for m in range(self.Args['M']+1):
            w = self.Args['w_m'+str(m)]
            dt = self.Args['dt']
            self.DataDev['MxSlv_cos(wdt)'+str(m)] = self.dev_arr(
              dtype=np.double, shape=(self.Args['Nr']-1, self.Args['Nx']))
            self.DataDev['MxSlv_sin(wdt)/w'+str(m)] = self.dev_arr(
              dtype=np.double, shape=(self.Args['Nr']-1, self.Args['Nx']))
            self.DataDev['MxSlv_1/w**2'+str(m)] = self.dev_arr(
              dtype=np.double, shape=(self.Args['Nr']-1, self.Args['Nx']))

            self.DataDev['MxSlv_cos(wdt)'+str(m)][:] = np.cos(w * dt)
            self.DataDev['MxSlv_sin(wdt)/w'+str(m)][:] = np.sin(w*dt) / w
            self.DataDev['MxSlv_1/w**2'+str(m)][:] = 1. / w**2
