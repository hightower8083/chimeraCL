import numpy as np

from pyopencl.array import zeros
from pyopencl.array import empty
from pyopencl import CommandQueue
from pyopencl import device_type
from pyopencl import create_some_context
from pyopencl import enqueue_barrier
from pyopencl import Program
from reikna.cluda import ocl_api

from chimeraCL import __path__ as src_path

src_path = src_path[0] + '/kernels/'

#compiler_options = ['-cl-fast-relaxed-math',]
compiler_options = []


class GenericMethodsCL:
    def init_generic_methods(self):
        self.set_global_working_group_size()

        generic_sources = []
        generic_sources.append(''.join(open(src_path+"generic.cl")\
                               .readlines()) )

        generic_sources = self.block_def_str + ''.join(generic_sources)

        prg = Program(self.ctx, generic_sources).\
            build(options=compiler_options)

        self._set_cdouble_to_zero_knl = prg.set_cdouble_to_zero
        self._cast_array_d2c_knl = prg.cast_array_d2c

    def set_global_working_group_size(self):
        if self.dev_type=='CPU':
            self.WGS = 32
        else:
            self.WGS = self.ctx.devices[0].max_work_group_size

        self.block_def_str = "#define BLOCK_SIZE {:d}\n".format(self.WGS)

    def get_wgs(self,Nelem):
        if Nelem <= self.WGS:
            return Nelem, Nelem
        else:
            WGS_tot = int(np.ceil(1.*Nelem/self.WGS))*self.WGS
            WGS = self.WGS
            return WGS, WGS_tot

    def dev_arr(self, val=None, shape=(1, ), dtype=np.double):
        if type(val) is np.ndarray:
            arr = self.thr.to_device(val)
        elif val==0:
            arr = zeros(self.queue, shape, dtype=dtype)
        else:
            arr = empty(self.queue, shape, dtype=dtype)
            if val is not None: arr.fill(val)
        return arr

    def send_args_to_dev(self):
        self.DataDev = {}
        for arg in self.Args.keys():
            arg_type = type(self.Args[arg])
            if arg_type is int:
                arg_dtype = np.uint32
            elif arg_type is float:
                arg_dtype = np.double
            elif arg_type is np.ndarray:
                arg_dtype = self.Args[arg].dtype

            self.DataDev[arg] = self.dev_arr(self.Args[arg],dtype=arg_dtype)

    def set_to_zero(self, arr):
        if arr.dtype != np.complex128:
            arr[:] = 0
            enqueue_barrier(self.queue)
        else:
            arr_size = arr.size
            WGS, WGS_tot = self.get_wgs(arr_size)
            self._set_cdouble_to_zero_knl(self.queue, (WGS_tot, ), (WGS, ),
                                          arr.data, np.uint32(arr_size)).wait()

    def cast_array_c2d(self,arr_in, arr_out):
        arr_size = arr_in.size
        WGS, WGS_tot = self.get_wgs(arr_size)
        self._cast_array_d2c_knl(self.queue,(WGS_tot,),(WGS,),
                                 arr_in.data, arr_out.data,
                                 np.uint32(arr_size)).wait()

class Communicator:
    def __init__(self, **ctx_kw_args):
        if ctx_kw_args == {}:
            print("Context is not chosen, please, do it now")
            print("(you can specify argument: answers=[..,] )")
            ctx_kw_args['interactive'] = True

        self.ctx = create_some_context(**ctx_kw_args)
        self.queue = CommandQueue(self.ctx)

        api = ocl_api()
        self.thr = api.Thread(cqd=self.queue)

        selected_dev = self.queue.device #ctx.devices[0]
        self.dev_type = device_type.to_string(selected_dev.type)
        self.plat_name = selected_dev.platform.vendor
        self.ocl_version = selected_dev.opencl_c_version

        print("{} device is chosen on {} platform with {}".
              format(self.dev_type,self.plat_name,self.ocl_version))

        if self.dev_type=='CPU' and self.plat_name=='Apple':
            print('\tReikna FFT is replaced by pyFFTW')
            self.fft_method = 'pyFFTW'
        else:
            self.fft_method = 'Reikna'

        if self.dev_type=='CPU':
            print('\tReikna MatrixMul is replaced by numpy.dot')
            self.dot_method = 'NumPy'
        else:
            self.dot_method = 'Reikna'

        self.sort_method = 'Radix'


