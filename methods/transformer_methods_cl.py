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

    def transform_field(self, arg_cmp, dir):
        # prepare the phase shift
        WGS, WGS_tot = self.get_wgs(self.Args['Nx'])
        phs_shft = self.dev_arr(dtype=np.complex128, shape=self.Args['Nx'])

        args = [phs_shft.data, self.DataDev['kx'].data,
                np.double(self.Args['Xmin']), np.uint32(self.Args['Nx']) ]
        self._phase_knl[dir](self.queue, (WGS_tot, ), (WGS, ), *args).wait()

        if dir == 0:
            dht_arg = 'DHT_m'
            arg_in = arg_cmp + '_m'
            arg_out = arg_cmp + '_fb_m'
            self._transform_forward(dht_arg,arg_in,arg_out,phs_shft)
        elif dir == 1:
            dht_arg = 'DHT_inv_m'
            arg_in = arg_cmp + '_fb_m'
            arg_out = arg_cmp + '_m'
            self._transform_backward(dht_arg,arg_in,arg_out,phs_shft)

    def _transform_forward(self, dht_arg,arg_in,arg_out,phs_shft):
        dir = 0
        WGS, WGS_tot = self.get_wgs(self.Args['NxNrm1'])

        # DHT of 0 mode into a temporal array
        buff1_dbl = self.DataDev[arg_in+'0'][1:].copy()
        buff2_dbl = self._dot0(self.DataDev[dht_arg+'0'], buff1_dbl)
        enqueue_barrier(self.queue)

        # FFT of 0 mode casted to complex dtype
        self.DataDev[arg_out+'0'][:] = buff2_dbl.astype(np.complex128)
        enqueue_barrier(self.queue)
        self.DataDev[arg_out+'0'] = self._fft(self.DataDev[arg_out+'0'],
                                              dir=dir)

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
            buff1_clx = self.DataDev[arg_in_m][1:,:].copy()

            self.DataDev[arg_out_m] = self._dot1(
                self.DataDev[dht_arg+str(m)], buff1_clx)
            enqueue_barrier(self.queue)

            # FFT of m mode
            self.DataDev[arg_out_m] = self._fft(self.DataDev[arg_out_m],
                                                dir=dir)

            # Phase shift of the result
            phs_str = [arg_out_m, 'NxNrm1', 'Nx']
            phs_args = [self.DataDev[arg].data for arg in phs_str]
            phs_args += [phs_shft.data, ]
            self._multiply_by_phase_knl(self.queue, (WGS_tot, ), (WGS, ),
                                        *phs_args).wait()

    def _transform_backward(self, dht_arg,arg_in,arg_out,phs_shft):
        dir = 1
        WGS, WGS_tot = self.get_wgs(self.Args['NxNrm1'])

        # Copy and phase-shift the field
        buff1_clx = self.DataDev[arg_in+'0'].copy()
        phs_str = ['NxNrm1', 'Nx']
        phs_args = [self.DataDev[arg].data for arg in phs_str]
        phs_args = [buff1_clx.data, ] + phs_args + [phs_shft.data, ]
        self._multiply_by_phase_knl(self.queue, (WGS_tot, ), (WGS, ),
                                    *phs_args).wait()

        # FFT of 0 mode and casting result to double dtype
        buff1_clx = self._fft(buff1_clx, dir=dir)
        buff1_dbl = self.dev_arr(dtype=np.double,
            shape=(self.Args['Nr']-1, self.Args['Nx']))
        self.cast_array_c2d(buff1_clx, buff1_dbl)

        # DHT of 0 mode
        buff2_dbl = self._dot0(self.DataDev[dht_arg+'0'], buff1_dbl)
        enqueue_barrier(self.queue)
        self.DataDev[arg_out+'0'][1:] = buff2_dbl

        # Same for m>0 modes
        for m in range(1, self.Args['M']+1):
            arg_in_m = arg_in + str(m)
            arg_out_m = arg_out + str(m)

            # Copy and phase-shift the field
            buff1_clx = self.DataDev[arg_in_m].copy()
            phs_str = ['NxNrm1', 'Nx']
            phs_args = [self.DataDev[arg].data for arg in phs_str]
            phs_args = [buff1_clx.data, ] + phs_args + [phs_shft.data, ]
            self._multiply_by_phase_knl(self.queue, (WGS_tot, ), (WGS, ),
                                        *phs_args).wait()

            # FFT of m mode into a temporal array
            buff1_clx = self._fft(buff1_clx, dir=dir)

            # DHT of m mode
            buff2_clx = self._dot1(self.DataDev[dht_arg+str(m)], buff1_clx)
            enqueue_barrier(self.queue)
            self.DataDev[arg_out_m][1:] = buff2_clx

    def _get_m1(self,fld_comp):
        if self.Args['M']==0:
            return

        buff = empty_like(self.DataDev[fld_comp + '_fb_m1'])

        arg_str = [fld_comp+'_fb_m1', 'Nx', 'NxNrm1',]
        args = [buff.data,] + [self.DataDev[arg].data for arg in arg_str]

        WGS, WGS_tot = self.get_wgs(self.Args['NxNrm1'])
        self._get_m1_knl(self.queue, (WGS_tot, ), (WGS, ),*args).wait()

        return buff

    def _prepare_dot(self):
        input_array_c = self.dev_arr(dtype=np.complex128,
                                     shape=(self.Args['Nr']-1, self.Args['Nx']))
        input_array_d = input_array_c.real

        input_transform = self.dev_arr(dtype=np.double,
                                       shape=(self.Args['Nr']-1,
                                              self.Args['Nr']-1))

        if self.comm.dot_method=='Reikna':
            self._dot0_knl = MatrixMul(input_transform, input_array_d,
                                       out_arr=input_array_d \
                                      ).compile(self.thr, fast_math=True)

            self._dot_knl = MatrixMul(input_transform, input_array_c,
                                      out_arr=input_array_c \
                                     ).compile(self.thr, fast_math=True)
            def dot0_wrp(a, b):
                c = empty_like(b)
                self._dot0_knl(c, a, b)
                return c

            def dot1_wrp(a, b):
                c = empty_like(b)
                self._dot_knl(c, a, b)
                return c

            self._dot0 = dot0_wrp
            self._dot1  = dot1_wrp

        elif self.comm.dot_method=='NumPy':
            def dot_wrp(a, b):
                c = np.dot(a.get(), b.get())
                c = to_device(self.queue, c)
                return c

            self._dot0 = dot_wrp
            self._dot1  = dot_wrp

    def _prepare_fft(self):
        input_array_c = self.dev_arr(dtype=np.complex128,
                                     shape=(self.Args['Nr']-1, self.Args['Nx']))
        if self.comm.fft_method=='pyFFTW':
            from pyfftw import empty_aligned,FFTW

            self.arr_fft_in = empty_aligned(input_array_c.shape,
                        dtype=np.complex128, n=16)
            self.arr_fft_out = empty_aligned(input_array_c.shape,
                        dtype=np.complex128, n=16)
            self._fft_knl = [FFTW(input_array=self.arr_fft_in,
                              output_array=self.arr_fft_out,
                              direction = 'FFTW_FORWARD', threads=4,axes=(1,)),
                         FFTW(input_array=self.arr_fft_in,
                              output_array=self.arr_fft_out,
                              direction = 'FFTW_BACKWARD', threads=4,axes=(1,))]
            def _fft(arr, dir=0):
                self.arr_fft_in[:] = arr.get()
                self._fft_knl[dir]()
                arr_out = to_device(self.queue,self.arr_fft_out)
                return arr_out

        elif self.comm.fft_method=='Reikna':
            fft = FFT(input_array_c, axes=(1,))
            self._fft_knl = fft.compile(self.thr)

            def _fft(arr, dir=0):
                arr_out = empty_like(arr)
                self._fft_knl(arr_out, arr,dir)
                return arr_out

        self._fft = _fft
