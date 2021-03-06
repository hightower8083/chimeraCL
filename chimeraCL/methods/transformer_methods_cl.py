import numpy as np

from pyopencl import Program
from pyopencl import enqueue_barrier
from pyopencl.array  import to_device, empty_like
from reikna.fft import FFT
from reikna.linalg import MatrixMul

from .generic_methods_cl import GenericMethodsCL
from .generic_methods_cl import compiler_options

from chimeraCL import __path__ as src_path
src_path = src_path[0] + '/kernels/'


class TransformerMethodsCL(GenericMethodsCL):
    def init_transformer_methods(self):
        self.init_generic_methods()
        self.set_global_working_group_size()

        transformer_sources = []
        transformer_sources.append(
            ''.join(open(src_path+"transformer_generic.cl").readlines()) )

        transformer_sources = self.block_def_str + ''.join(transformer_sources)

        prg = Program(self.ctx, transformer_sources).\
            build(options=compiler_options)

        self._phase_knl = {0: prg.get_phase_minus,
                           1: prg.get_phase_plus}
        self._multiply_by_phase_knl = prg.multiply_by_phase
        self._get_m1_knl = prg.get_m1

        self._prepare_fft()
        self._prepare_dot()

    def transform_field(self, arg_cmp, dir, mode):
        # do the phase shift
        WGS, WGS_tot = self.get_wgs(self.Args['Nx'])

        args = [self.DataDev['phs_shft'].data, self.DataDev['kx'].data,
                np.double(self.Args['Xmin']), np.uint32(self.Args['Nx']) ]
        self._phase_knl[dir](self.queue, (WGS_tot, ), (WGS, ), *args).wait()

        if mode=='full':
            forward = self._transform_forward
            backward = self._transform_backward
        elif mode=='half':
            forward = self._half_transform_forward
            backward = self._half_transform_backward

        if dir == 0:
            dht_arg = 'DHT_m'
            arg_in = arg_cmp + '_m'
            arg_out = arg_cmp + '_fb_m'
            transformer = forward
        elif dir == 1:
            dht_arg = 'DHT_inv_m'
            arg_in = arg_cmp + '_fb_m'
            arg_out = arg_cmp + '_m'
            transformer = backward

        transformer(dht_arg, arg_in, arg_out, self.DataDev['phs_shft'])

    def field_poiss_vec(self, fld):
        for m in range(0,self.Args['M']+1):
            for comp in self.Args['vec_comps']:
                fld_arg = fld + comp + '_fb_m' + str(m)
                self.mult_elementwise(self.DataDev['Poiss_m'+ str(m)],
                                      self.DataDev[fld_arg])

    def field_poiss_scl(self, fld):
        for m in range(0,self.Args['M']+1):
            fld_arg = fld + '_fb_m' + str(m)
            self.mult_elementwise(self.DataDev['Poiss_m'+ str(m)],
                                  self.DataDev[fld_arg])

    def fields_smooth(self, flds):
        for m in range(0,self.Args['M']+1):
            for fld in flds:
                fld_arg = fld + '_fb_m' + str(m)
                self.mult_elementwise(
                    self.DataDev['SmoothingFilter_m'+ str(m)],
                    self.DataDev[fld_arg])

    def field_grad(self, scl_in, vec_out):

        self._get_mm1_scl(scl_in)

        for m in range(0,self.Args['M']+1):
            fld_x_m_out = vec_out + 'x' + '_fb_m' + str(m)
            fld_y_m_out = vec_out + 'y' + '_fb_m' + str(m)
            fld_z_m_out = vec_out + 'z' + '_fb_m' + str(m)

            for fld in (fld_x_m_out, fld_y_m_out, fld_z_m_out):
                self.set_to(self.DataDev[fld], 0.)

            self.ab_dot_x(1.j, self.DataDev['kx'],
                          self.DataDev[scl_in+'_fb_m'+str(m)],
                          self.DataDev[fld_x_m_out])

            # m-1 cmoponents
            if m > 0:
                fld_scl = scl_in + '_fb_m' + str(m-1)
            elif self.Args['M'] > 0:
                fld_scl = 'buff_fb_m-1_x'
            else:
                continue

            self._cdot(self.DataDev['fld_buff0_c'],
                       self.DataDev['dDHT_minus_m'+str(m)],
                       self.DataDev[fld_scl] )

            self.zpaxz(self.DataDev[fld_y_m_out],
                       -1., self.DataDev['fld_buff0_c'])

            self.zpaxz(self.DataDev[fld_z_m_out],
                       -1.j, self.DataDev['fld_buff0_c'])

            # m+1 cmoponents
            if m < self.Args['M']:
                fld_scl = scl_in + '_fb_m' + str(m+1)

                self._cdot(self.DataDev['fld_buff0_c'],
                           self.DataDev['dDHT_plus_m'+str(m)],
                           self.DataDev[fld_scl] )

                self.append_c2c(self.DataDev[fld_y_m_out],
                                self.DataDev['fld_buff0_c'])

                self.zpaxz(self.DataDev[fld_z_m_out],
                           -1.j, self.DataDev['fld_buff0_c'])

    def field_div(self, vec_in, scl_out):

        for comp in ['y', 'z']:
            self._get_mm1_scl(vec_in+comp, comp)

        for m in range(0,self.Args['M']+1):
            fld_m_out = scl_out + '_fb_m' + str(m)
            self.set_to(self.DataDev[fld_m_out], 0.)

            self.ab_dot_x(1.j, self.DataDev['kx'],
                          self.DataDev[vec_in + 'x' +'_fb_m'+str(m)],
                          self.DataDev[fld_m_out])

            # m-1 cmoponents
            if m > 0:
                fld_y = vec_in + 'y' + '_fb_m' + str(m-1)
                fld_z = vec_in + 'z' + '_fb_m' + str(m-1)
            elif self.Args['M'] > 0:
                fld_y = 'buff_fb_m-1_y'
                fld_z = 'buff_fb_m-1_z'
            else:
                continue

            self.axpbyz(-1.j, self.DataDev[fld_z],
                        -1.0, self.DataDev[fld_y],
                        self.DataDev['fld_buff0_c'])

            self._cdot(self.DataDev['fld_buff1_c'],
                       self.DataDev['dDHT_minus_m'+str(m)],
                       self.DataDev['fld_buff0_c'] )

            self.append_c2c(self.DataDev[fld_m_out],
                            self.DataDev['fld_buff1_c'])

            # m+1 cmoponents
            if m < self.Args['M']:
                fld_y = vec_in + 'y' + '_fb_m' + str(m+1)
                fld_z = vec_in + 'z' + '_fb_m' + str(m+1)

                self.axpbyz(-1.j, self.DataDev[fld_z],
                            1.0, self.DataDev[fld_y],
                            self.DataDev['fld_buff0_c'])

                self._cdot(self.DataDev['fld_buff1_c'],
                           self.DataDev['dDHT_plus_m'+str(m)],
                           self.DataDev['fld_buff0_c'] )

                self.append_c2c(self.DataDev[fld_m_out],
                                self.DataDev['fld_buff1_c'])

    def field_rot(self, fld_in, fld_out):

        self._get_mm1_vec(fld_in)

        for m in range(0,self.Args['M']+1):
            fld_x_m_out = fld_out + 'x' + '_fb_m' + str(m)
            fld_y_m_out = fld_out + 'y' + '_fb_m' + str(m)
            fld_z_m_out = fld_out + 'z' + '_fb_m' + str(m)

            for fld in (fld_x_m_out, fld_y_m_out, fld_z_m_out):
                self.set_to(self.DataDev[fld], 0.0)

            self.ab_dot_x(-1.j, self.DataDev['kx'],
                          self.DataDev[fld_in+'z'+'_fb_m'+str(m)],
                          self.DataDev[fld_y_m_out])

            self.ab_dot_x(1.j, self.DataDev['kx'],
                          self.DataDev[fld_in+'y'+'_fb_m'+str(m)],
                          self.DataDev[fld_z_m_out])

            # m-1 cmoponents
            if m > 0:
                fld_x = fld_in + 'x' + '_fb_m' + str(m-1)
                fld_y = fld_in + 'y' + '_fb_m' + str(m-1)
                fld_z = fld_in + 'z' + '_fb_m' + str(m-1)
            elif self.Args['M'] > 0:
                fld_x = 'buff_fb_m-1_x'
                fld_y = 'buff_fb_m-1_y'
                fld_z = 'buff_fb_m-1_z'
            else:
                continue

            self.axpbyz(-1, self.DataDev[fld_z],
                        1.j, self.DataDev[fld_y],
                        self.DataDev['fld_buff0_c'])

            self._cdot(self.DataDev['fld_buff1_c'],
                       self.DataDev['dDHT_minus_m'+str(m)],
                       self.DataDev['fld_buff0_c'] )

            self.append_c2c(self.DataDev[fld_x_m_out],
                            self.DataDev['fld_buff1_c'])

            self._cdot(self.DataDev['fld_buff0_c'],
                       self.DataDev['dDHT_minus_m'+str(m)],
                       self.DataDev[fld_x] )

            self.zpaxz(self.DataDev[fld_y_m_out],
                       -1.j, self.DataDev['fld_buff0_c'])

            self.zpaxz(self.DataDev[fld_z_m_out],
                       1, self.DataDev['fld_buff0_c'])

            # m+1 cmoponents
            if m < self.Args['M']:
                fld_x = fld_in + 'x' + '_fb_m' + str(m+1)
                fld_y = fld_in + 'y' + '_fb_m' + str(m+1)
                fld_z = fld_in + 'z' + '_fb_m' + str(m+1)

                self.axpbyz(1, self.DataDev[fld_z],
                            1.j, self.DataDev[fld_y],
                            self.DataDev['fld_buff0_c'])

                self._cdot(self.DataDev['fld_buff1_c'],
                           self.DataDev['dDHT_plus_m'+str(m)],
                           self.DataDev['fld_buff0_c'] )

                self.append_c2c(self.DataDev[fld_x_m_out],
                                self.DataDev['fld_buff1_c'])

                self._cdot(self.DataDev['fld_buff0_c'],
                           self.DataDev['dDHT_plus_m'+str(m)],
                           self.DataDev[fld_x] )

                self.zpaxz(self.DataDev[fld_y_m_out],
                           -1.j, self.DataDev['fld_buff0_c'])

                self.zpaxz(self.DataDev[fld_z_m_out],
                           -1, self.DataDev['fld_buff0_c'])

    def _get_mm1_vec(self, fld):
        if self.Args['M']==0:
            return

        for comp in self.Args['vec_comps']:
            arg_str = [fld + comp + '_fb_m1', 'Nx', 'NxNrm1',]
            args = [self.DataDev['buff_fb_m-1_' + comp].data,] \
                   + [self.DataDev[arg].data for arg in arg_str]

            WGS, WGS_tot = self.get_wgs(self.Args['NxNrm1'])
            self._get_m1_knl(self.queue, (WGS_tot, ), (WGS, ),*args).wait()

    def _get_mm1_scl(self, fld, comp='x'):
        # Note: for scalar the X component of buffer will be used

        if self.Args['M']==0:
            return

        arg_str = [fld + '_fb_m1', 'Nx', 'NxNrm1',]
        args = [self.DataDev['buff_fb_m-1_' + comp].data,] \
               + [self.DataDev[arg].data for arg in arg_str]

        WGS, WGS_tot = self.get_wgs(self.Args['NxNrm1'])
        self._get_m1_knl(self.queue, (WGS_tot, ), (WGS, ),*args).wait()

    def _transform_forward(self, dht_arg, arg_in, arg_out, phs_shft):
        dir = 0
        WGS, WGS_tot = self.get_wgs(self.Args['NxNrm1'])

        # DHT of 0 mode into a temporal array
        self.DataDev['fld_buff0_d'][:] = self.DataDev[arg_in+'0'][1:]
        self._ddot(self.DataDev['fld_buff1_d'], self.DataDev[dht_arg+'0'],
                   self.DataDev['fld_buff0_d'])
        enqueue_barrier(self.queue)

        # FFT of 0 mode casted to complex dtype
        self.DataDev[arg_out+'0'][:] = self.DataDev['fld_buff1_d'].\
                                        astype(np.complex128)
        enqueue_barrier(self.queue)
        self._fft(self.DataDev[arg_out+'0'], self.DataDev[arg_out+'0'], dir)

        # Phase shift of the result
        phs_str = [arg_out+'0', 'NxNrm1', 'Nx']
        phs_args = [self.DataDev[arg].data for arg in phs_str]
        phs_args += [phs_shft.data, ]
        self._multiply_by_phase_knl(self.queue, (WGS_tot, ), (WGS,),
                                    *phs_args).wait()

        # Same for m>0 modes
        for m in range(1,self.Args['M']+1):
            arg_in_m = arg_in + str(m)
            arg_out_m = arg_out + str(m)

            # DHT of m mode into a temporal array
            self.DataDev['fld_buff0_c'][:] = self.DataDev[arg_in_m][1:]
            self._cdot(self.DataDev[arg_out_m], self.DataDev[dht_arg+str(m)],
                       self.DataDev['fld_buff0_c'])
            enqueue_barrier(self.queue)

            # FFT of m mode
            self._fft(self.DataDev[arg_out_m], self.DataDev[arg_out_m], dir)

            # Phase shift of the result
            phs_str = [arg_out_m, 'NxNrm1', 'Nx']
            phs_args = [self.DataDev[arg].data for arg in phs_str]
            phs_args += [phs_shft.data, ]
            self._multiply_by_phase_knl(self.queue, (WGS_tot, ), (WGS, ),
                                        *phs_args).wait()

    def _transform_backward(self, dht_arg, arg_in, arg_out, phs_shft):
        dir = 1
        WGS, WGS_tot = self.get_wgs(self.Args['NxNrm1'])

        # Copy and phase-shift the field
        self.DataDev['fld_buff0_c'][:] = self.DataDev[arg_in+'0']

        phs_str = ['NxNrm1', 'Nx']
        phs_args = [self.DataDev[arg].data for arg in phs_str]
        phs_args = [self.DataDev['fld_buff0_c'].data, ] + phs_args \
                   + [phs_shft.data, ]

        self._multiply_by_phase_knl(self.queue, (WGS_tot, ), (WGS, ),
                                    *phs_args).wait()

        # FFT of 0 mode and casting result to double dtype
        self._fft(self.DataDev['fld_buff0_c'], self.DataDev['fld_buff0_c'], dir)
        self.cast_array_c2d(self.DataDev['fld_buff0_c'],
                            self.DataDev['fld_buff0_d'])

        # DHT of 0 mode
        self._ddot(self.DataDev['fld_buff1_d'], self.DataDev[dht_arg+'0'],
                   self.DataDev['fld_buff0_d'])
        enqueue_barrier(self.queue)
        self.DataDev[arg_out+'0'][1:] = self.DataDev['fld_buff1_d']

        # Same for m>0 modes
        for m in range(1, self.Args['M']+1):
            arg_in_m = arg_in + str(m)
            arg_out_m = arg_out + str(m)

            # Copy and phase-shift the field
            self.DataDev['fld_buff0_c'][:] = self.DataDev[arg_in_m]
            phs_str = ['NxNrm1', 'Nx']
            phs_args = [self.DataDev[arg].data for arg in phs_str]
            phs_args = [self.DataDev['fld_buff0_c'].data, ] + phs_args \
                       + [phs_shft.data, ]
            self._multiply_by_phase_knl(self.queue, (WGS_tot, ), (WGS, ),
                                        *phs_args).wait()

            # FFT of m mode into a temporal array
            self._fft(self.DataDev['fld_buff0_c'], self.DataDev['fld_buff0_c'],
                      dir)

            # DHT of m mode
            self._cdot(self.DataDev['fld_buff1_c'],
                       self.DataDev[dht_arg+str(m)],
                       self.DataDev['fld_buff0_c'])
            enqueue_barrier(self.queue)
            self.DataDev[arg_out_m][1:] = self.DataDev['fld_buff1_c']

    def _half_transform_backward(self, dht_arg, arg_in, arg_out, phs_shft):
        dir = 1
        WGS, WGS_tot = self.get_wgs(self.Args['NxNrm1'])

        # Copy and phase-shift the field
        self.DataDev['fld_buff0_c'][:] = self.DataDev[arg_in+'0']

        phs_str = ['NxNrm1', 'Nx']
        phs_args = [self.DataDev[arg].data for arg in phs_str]
        phs_args = [self.DataDev['fld_buff0_c'].data, ] + phs_args \
                   + [phs_shft.data, ]

        self._multiply_by_phase_knl(self.queue, (WGS_tot, ), (WGS, ),
                                    *phs_args).wait()

        # FFT of 0 mode and casting result to double dtype
        self._fft(self.DataDev['fld_buff0_c'], self.DataDev['fld_buff0_c'], dir)
        self.cast_array_c2d(self.DataDev['fld_buff0_c'],
                            self.DataDev['fld_buff0_d'])

        self.DataDev[arg_out+'0'][1:] = self.DataDev['fld_buff0_d']

        # Same for m>0 modes
        for m in range(1, self.Args['M']+1):
            arg_in_m = arg_in + str(m)
            arg_out_m = arg_out + str(m)

            # Copy and phase-shift the field
            self.DataDev['fld_buff0_c'][:] = self.DataDev[arg_in_m]
            phs_str = ['NxNrm1', 'Nx']
            phs_args = [self.DataDev[arg].data for arg in phs_str]
            phs_args = [self.DataDev['fld_buff0_c'].data, ] + phs_args \
                       + [phs_shft.data, ]
            self._multiply_by_phase_knl(self.queue, (WGS_tot, ), (WGS, ),
                                        *phs_args).wait()

            # FFT of m mode into a temporal array
            self._fft(self.DataDev['fld_buff0_c'], self.DataDev['fld_buff0_c'],
                      dir)

            self.DataDev[arg_out_m][1:] = self.DataDev['fld_buff0_c']

    def _half_transform_forward(self, dht_arg, arg_in, arg_out, phs_shft):
        dir = 0
        WGS, WGS_tot = self.get_wgs(self.Args['NxNrm1'])

        self.DataDev[arg_out+'0'][:] = self.DataDev[arg_in+'0'][1:].\
                                        astype(np.complex128)
        self._fft(self.DataDev[arg_out+'0'], self.DataDev[arg_out+'0'], dir)

        # Phase shift of the result
        phs_str = [arg_out+'0', 'NxNrm1', 'Nx']
        phs_args = [self.DataDev[arg].data for arg in phs_str]
        phs_args += [phs_shft.data, ]
        self._multiply_by_phase_knl(self.queue, (WGS_tot, ), (WGS,),
                                    *phs_args).wait()

        # Same for m>0 modes
        for m in range(1,self.Args['M']+1):
            arg_in_m = arg_in + str(m)
            arg_out_m = arg_out + str(m)

            self.DataDev[arg_out_m][:] = self.DataDev[arg_in_m][1:]
            self._fft(self.DataDev[arg_out_m], self.DataDev[arg_out_m], dir)

            # Phase shift of the result
            phs_str = [arg_out_m, 'NxNrm1', 'Nx']
            phs_args = [self.DataDev[arg].data for arg in phs_str]
            phs_args += [phs_shft.data, ]
            self._multiply_by_phase_knl(self.queue, (WGS_tot, ), (WGS, ),
                                        *phs_args).wait()


    def _prepare_dot(self):
        input_transform = self.dev_arr(dtype=np.double,
                                       shape=(self.Args['Nr']-1,
                                              self.Args['Nr']-1))

        if self.comm.dot_method=='Reikna':
            self._ddot = MatrixMul(input_transform,
                                   self.DataDev['fld_buff0_d'],
                                   out_arr=self.DataDev['fld_buff1_d'] \
                                  ).compile(self.thr, fast_math=True)

            self._cdot = MatrixMul(input_transform,
                                   self.DataDev['fld_buff0_c'],
                                   out_arr=self.DataDev['fld_buff1_c'], \
                                  ).compile(self.thr, fast_math=True)

        elif self.comm.dot_method=='NumPy':
            def dot_wrp(c, a, b):
                c_host = np.dot(a.get(), b.get())
                c[:] = to_device(self.queue, c_host)

            self._ddot = dot_wrp
            self._cdot  = dot_wrp

    def _prepare_fft(self):
        if self.comm.fft_method=='pyFFTW':
            from pyfftw import empty_aligned, FFTW

            target_shape = self.DataDev['fld_buff0_c'].shape

            self.arr_fft_in = empty_aligned(target_shape,
                                            dtype=np.complex128, n=16)
            self.arr_fft_out = empty_aligned(target_shape,
                                             dtype=np.complex128, n=16)
            self._fft_knl = [FFTW(input_array=self.arr_fft_in,
                              output_array=self.arr_fft_out,
                              direction = 'FFTW_FORWARD', threads=4,
                              axes=(1, )),
                             FFTW(input_array=self.arr_fft_in,
                              output_array=self.arr_fft_out,
                              direction = 'FFTW_BACKWARD', threads=4,
                              axes=(1, ))]
            def _fft(arr_out, arr, dir):
                self.arr_fft_in[:] = arr.get()
                self._fft_knl[dir]()
                arr_out[:] = to_device(self.queue, self.arr_fft_out)
                return arr_out
            self._fft = _fft

        elif self.comm.fft_method=='Reikna':
            fft = FFT(self.DataDev['fld_buff0_c'], axes=(1,))
            self._fft = fft.compile(self.thr)
