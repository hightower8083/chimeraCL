import numpy as np

from pyopencl import enqueue_barrier
from pyopencl.array  import to_device, empty_like
from reikna.fft import FFT
from reikna.linalg import MatrixMul

class TransformerMethodsCL():
    def init_transformer_methods(self):
        self._prepare_fft()
        self._prepare_dot()

    def transform_field(self, arg,dir):
        if dir == 0:
            dht_arg = 'DHT_m'
            arg_in = arg + '_m'
            arg_out = arg + '_fb_m'

            self.DataDev['field_fb_aux1_dbl'] = self.DataDev[arg_in+'0']\
                                                  [1:].copy()
            self.DataDev['field_fb_aux2_dbl'] = self._dot0(
                self.DataDev[dht_arg+'0'], self.DataDev['field_fb_aux1_dbl'])

            enqueue_barrier(self.queue)

            self.DataDev[arg_out+'0'][:] = self.DataDev['field_fb_aux2_dbl']\
                                                   .astype(np.complex128)
            enqueue_barrier(self.queue)
            self.DataDev[arg_out+'0'] = self._fft(self.DataDev[arg_out+'0'],
                                                  dir=dir)

            for m in range(1,self.Args['M']+1):
                arg_in_m = arg_in + str(m)
                arg_out_m = arg_out + str(m)

                self.DataDev['field_fb_aux1_clx'] = self.DataDev[arg_in_m]\
                                                      [1:,:].copy()

                self.DataDev[arg_out_m] = self._dot1(
                    self.DataDev[dht_arg+str(m)],
                    self.DataDev['field_fb_aux1_clx'])

                enqueue_barrier(self.queue)
                self.DataDev[arg_out_m] = self._fft(self.DataDev[arg_out_m],
                                                    dir=dir)
        elif dir == 1:
            dht_arg = 'DHT_inv_m'
            arg_in = arg + '_fb_m'
            arg_out = arg + '_m'

            self.DataDev['field_fb_aux1_clx'] = self._fft(
                self.DataDev[arg_in+'0'], dir=dir)

            self.cast_array_c2d(self.DataDev['field_fb_aux1_clx'],
                                 self.DataDev['field_fb_aux1_dbl'])

            self.DataDev['field_fb_aux2_dbl'] = \
                self._dot0(self.DataDev[dht_arg+'0'],
                           self.DataDev['field_fb_aux1_dbl'])

            enqueue_barrier(self.queue)
            self.DataDev[arg_out+'0'][1:] = self.DataDev['field_fb_aux2_dbl']

            for m in range(1, self.Args['M']+1):
                arg_in_m = arg_in + str(m)
                arg_out_m = arg_out + str(m)
                self.DataDev['field_fb_aux1_clx'] = self._fft(
                    self.DataDev[arg_in_m], dir=dir)

                self.DataDev['field_fb_aux2_clx'] = self._dot1(
                    self.DataDev[dht_arg+str(m)],
                    self.DataDev['field_fb_aux1_clx'])

                enqueue_barrier(self.queue)
                self.DataDev[arg_out_m][1:] = self.DataDev['field_fb_aux2_clx']

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
