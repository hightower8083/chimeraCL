import numpy as np
from scipy.special import jn_zeros, jn

from chimeraCL.methods.transformer_methods_cl import TransformerMethodsCL


class Transformer(TransformerMethodsCL):
    def init_transformer(self):
        """
        Initialize the data for Fourier-Bessel transformations
        """
        self._init_transformer_data_on_dev()
        self.init_transformer_methods()
        self._make_spectral_axes()
        self._make_DHT()

    def fb_transform(self, scals=[], vects=[], dir=0, mode='full'):
        """
        Warper for the Fourier-Bessel transforms of the groups
        of vectors and scalars
        """
        for sclr in scals:
            self.transform_field(sclr, dir=dir, mode=mode)
        for vect in vects:
            for comp in self.Args['vec_comps']:
                self.transform_field(vect+comp, dir=dir, mode=mode)

    def _make_spectral_axes(self):
        """
        Prepare the grids in the Fourier and Bessel spectral space
        """

        dx = self.Args['dx']
        Nx = self.Args['Nx']
        dr = self.Args['dr']
        Nr = self.Args['Nr']

        kx = 2*np.pi*np.fft.fftfreq(Nx, dx)

        R_period = self.Args['Rgrid'][-1] + 0.5*dr

        self.Args['kx'] = kx

        for m in range(self.Args['M']+2):
            self.Args['kr_m'+str(m)] = jn_zeros(m, Nr-1) / R_period

        for m in range(self.Args['M']+1):
            kr = self.Args['kr_m'+str(m)]
            self.Args['w_m'+str(m)] = np.sqrt(kx[None,:]**2 + kr[:,None]**2)

            self.Args['Poiss_m'+str(m)] = 1./self.Args['w_m'+str(m)]**2

        for m in range(self.Args['M']+1):
            kr = self.Args['kr_m'+str(m)]
            self.Args['SmoothingFilter_m'+str(m)] = \
                (1 - np.sin(0.5*np.pi*kx[None,:]/kx.max())**4) \
              * (1 - np.sin(0.5*np.pi*kr[:,None]/kr.max())**4)

        for m in range(self.Args['M']+2):
            self.Args['dont_send'].append('kr_m'+str(m))
            self.Args['dont_send'].append('w_m'+str(m))
            self.Args['dont_keep'].append('Poiss_m'+str(m))
            self.Args['dont_keep'].append('SmoothingFilter_m'+str(m))

    def _make_DHT(self):
        """
        Prepare the matricies for the forward and backward
        Hankel transforms and differential operations in the
        spectral space
        """

        Rgrid = self.Args['Rgrid'][1:,None]
        R_period = Rgrid[-1] + 0.5*self.Args['dr']

        for m in range(self.Args['M']+1):
            # make spectral axes for the more m and satellite mode m+1 and m-1
            kr_0 = jn_zeros(m,   self.Args['Nr']-1) / R_period
            kr_p = jn_zeros(m+1, self.Args['Nr']-1) / R_period
            kr_m = jn_zeros(m-1, self.Args['Nr']-1) / R_period

            # make the backward Hankel transform
            self.Args['DHT_inv_m'+str(m)] = jn(m, Rgrid * kr_0)

            # make the forward Hankel transform by inversing the backward one
            self.Args['DHT_m'+str(m)] = np.linalg.pinv(
                self.Args['DHT_inv_m'+str(m)])

            # make spectral differential operator matrices
            self.Args['dDHT_plus_m'+str(m)] = self.Args['DHT_m'+str(m)].dot(
                0.5 * kr_p * jn(m, Rgrid * kr_p))
            self.Args['dDHT_minus_m'+str(m)] = self.Args['DHT_m'+str(m)].dot(
                0.5 * kr_m * jn(m, Rgrid * kr_m))

            self.Args['dont_keep'].append('DHT_inv_m'+str(m))
            self.Args['dont_keep'].append('DHT_m'+str(m))
            self.Args['dont_keep'].append('dDHT_plus_m'+str(m))
            self.Args['dont_keep'].append('dDHT_minus_m'+str(m))


    def _init_transformer_data_on_dev(self):
        # list the names of all scalar and vector fields components
        flds_str = ['E', 'B', 'G', 'J', 'dN0', 'dN1']
        flds_comps_str = ['rho',]
        for fld_str in flds_str:
            for comp_str in self.Args['vec_comps']:
                flds_comps_str.append(fld_str + comp_str)

        # allocate the fields arrays
        for arg in flds_comps_str:
            arg += '_fb_m'
            for m in range(self.Args['M']+1):
                self.DataDev[arg+str(m)] = self.dev_arr(
                    val=0, dtype=np.complex128,
                    shape=(self.Args['Nr']-1, self.Args['Nx']))

        # allocate the buffer for the mode m=-1
        for comp in self.Args['vec_comps']:
            self.DataDev['buff_fb_m-1_' + comp] = self.dev_arr(val=0,
                dtype=np.complex128, shape=(self.Args['Nr']-1, self.Args['Nx']))

        # allocate the buffer to keep phase shift data exp(i*kx*x0)
        self.DataDev['phs_shft'] = self.dev_arr(dtype=np.complex128,val=0,
                                                shape=self.Args['Nx'])

        # allocate the auxilary field buffers
        buff_dtypes = {'d':np.double, 'c':np.complex128}
        for buff_i in range(2):
            for buff_dtype in buff_dtypes.keys():
                arg_str = '_'.join(('fld', 'buff'+str(buff_i), buff_dtype))
                self.DataDev[arg_str] = self.dev_arr(val=0,
                    shape=(self.Args['Nr']-1, self.Args['Nx']),
                    dtype=buff_dtypes[buff_dtype] )
