import numpy as np
from scipy.special import jn_zeros, jn

from chimeraCL.methods.transformer_methods_cl import TransformerMethodsCL


class Transformer(TransformerMethodsCL):
    def init_transformer(self):
        self.init_transformer_methods()
        self._make_spectral_axes()
        self._make_DHT()
        self._init_data_on_dev()

    def fb_transform(self, comps=[], dir=0):
        for comp in comps:
            self.transform_field(comp, dir=dir)

    def _make_spectral_axes(self):
        if 'KxShift' in self.Args:
            self.Args['kx0'] = 2*np.pi*self.Args['KxShift']
        else:
            self.Args['kx0'] = 0.0

        self.Args['kx_env'] = 2*np.pi*np.fft.fftfreq(self.Args['Nx'],
                                                     self.Args['dx'])

        self.Args['kx'] = self.Args['kx0'] + self.Args['kx_env']
        self.Args['dkx'] = (self.Args['kx'][1]-self.Args['kx'][0]) / (2*np.pi)

        for m in range(self.Args['M']+2):
            self.Args['kr_m'+str(m)] = jn_zeros(m, self.Args['Nr']-1) / \
                                                   self.Args['R_period']
        for m in range(self.Args['M']+1):
            self.Args['w_m'+str(m)] = np.sqrt(
                self.Args['kx'][None,:]**2
                + self.Args['kr_m'+str(m)][:,None]**2)

    def _make_DHT(self):
        for m in range(self.Args['M']+1):
            self.Args['DHT_inv_m'+str(m)] = jn(m, \
                self.Args['Rgrid'][1:,None] * self.Args['kr_m'+str(m)][None,:])
            self.Args['DHT_m'+str(m)] = np.linalg.inv(
                self.Args['DHT_inv_m'+str(m)])

            self.Args['dDHT_plus_m'+str(m)] = np.eye(self.Args['Nr']-1)
            self.Args['dDHT_minus_m'+str(m)] = np.eye(self.Args['Nr']-1)

    def _init_data_on_dev(self):
        args_fld_init = ['rho', 'Jx', 'Jy', 'Jz',
                         'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']

        for arg in args_fld_init:
            arg += '_fb_m'
            for m in range(self.Args['M']+1):
                self.DataDev[arg+str(m)] = self.dev_arr(
                    val=0, dtype=np.complex128,
                    shape=(self.Args['Nr']-1, self.Args['Nx']))
