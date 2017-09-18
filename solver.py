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
        """
        kx_g,w = self.Args['kx_g'], self.Args['w']
        kx_g = kx_g[:,:,None]
        dt = self.Args['TimeStep']
        for m in range(self.Args['M']+1):
            self.DataDev['MxSlv_E_0_m'+str(m)]  = self.dev_arr(
              dtype=np.double, shape=(self.Args['Nr']-1, self.Args['Nx']))
            self.DataDev['MxSlv_E_1_m'+str(m)]  = self.dev_arr(
              dtype=np.double, shape=(self.Args['Nr']-1, self.Args['Nx']))
            self.DataDev['MxSlv_E_2_m'+str(m)]  = self.dev_arr(
              dtype=np.double, shape=(self.Args['Nr']-1, self.Args['Nx']))
            self.DataDev['MxSlv_G_0_m'+str(m)]  = self.dev_arr(
              dtype=np.double, shape=(self.Args['Nr']-1, self.Args['Nx']))
            self.DataDev['MxSlv_G_1_m'+str(m)] = \
              self.DataDev['MxSlv_E_0_m'+str(m)]


            self.DataDev['MxSlv_E_0_m'+str(m)][:] = \
              np.cos(self.Args['w_m'+str(m)] * dt)
            self.DataDev['MxSlv_E_1_m'+str(m)][:] = \
              np.sin(self.Args['w_m'+str(m)] * dt) / self.Args['w_m'+str(m)]
            self.DataDev['MxSlv_G_0_m'+str(m)][:] = \
              -self.Args['w_m'+str(m)] * np.sin(self.Args['w_m'+str(m)] * dt)
            self.DataDev['MxSlv_G_1_m'+str(m)] = \
              self.DataDev['MxSlv_E_0_m'+str(m)]


            self.Data['PSATD_E'][:,:,:,2] = -np.sin(dt*w) / w


            self.Data['PSATD_E'][:,:,:,3] = (dt*w*np.cos(dt*w)-np.sin(dt*w))\
              / w**3 / dt
            self.Data['PSATD_E'][:,:,:,4] = (np.sin(dt*w)-dt*w)/w**3/dt

            self.Data['PSATD_G'][:,:,:,2] = 1 - np.cos(dt*w)
            self.Data['PSATD_G'][:,:,:,3] = (1-np.cos(dt*w)-dt*w*np.sin(dt*w))\
              / w**2 / dt
            self.Data['PSATD_G'][:,:,:,4] = (np.cos(dt*w)-1) / w**2 / dt
        """
        pass
