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

compiler_options = ['-cl-fast-relaxed-math',]
#compiler_options = []


class GenericMethodsCL:
    def init_generic_methods(self):
        self.set_global_working_group_size()

        generic_sources = []
        generic_sources.append(''.join(open(src_path+"generic.cl")\
                               .readlines()) )

        generic_sources = self.block_def_str + ''.join(generic_sources)

        prg = Program(self.ctx, generic_sources).\
            build(options=compiler_options)

        self._cast_array_d2c_knl = prg.cast_array_d2c
        self._axpbyz_c2c_knl = prg.axpbyz_c2c
        self._zpaxz_c2c_knl = prg.zpaxz_c2c
        self._ab_dot_x_knl =  prg.ab_dot_x
        self._append_c2c_knl = prg.append_c2c
        self._set_cdouble_to_knl = prg.set_cdouble_to
        self._mult_elementwise_knl = prg.mult_elementwise_d2c

    def set_global_working_group_size(self):
        if self.dev_type=='CPU':
            self.WGS = 32
        else:
            self.WGS = 512 #self.ctx.devices[0].max_work_group_size

        self.block_def_str = "#define BLOCK_SIZE {:d}\n".format(self.WGS)

    def get_wgs(self,Nelem):
        if Nelem <= self.WGS:
            return Nelem, Nelem
        else:
            WGS_tot = int(np.ceil(1.*Nelem/self.WGS))*self.WGS
            WGS = self.WGS
            return WGS, WGS_tot

    def send_args_to_dev(self):
        original_keys = list(self.Args.keys())
        for arg in original_keys:
            arg_type = type(self.Args[arg])
            if 'dont_send' in self.Args.keys():
                if arg in self.Args['dont_send']:
                    continue

            if arg_type is int:
                arg_dtype = np.uint32
            elif arg_type is float:
                arg_dtype = np.double
            elif arg_type is np.ndarray:
                arg_dtype = self.Args[arg].dtype
            else:
                continue

            self.DataDev[arg] = self.dev_arr(self.Args[arg],dtype=arg_dtype)
            if 'dont_keep' in self.Args.keys():
                if arg in self.Args['dont_keep']:
                    self.Args.pop(arg);

    def dev_arr(self, val=None, shape=(1, ), dtype=np.double, allocator=None):
        if type(val) is np.ndarray:
            arr = self.thr.to_device(val)
        elif val==0:
            arr = zeros(self.queue, shape, dtype=dtype, allocator=allocator)
        else:
            arr = empty(self.queue, shape, dtype=dtype, allocator=allocator)
            if val is not None:
                self.set_to(arr, val)
        return arr

    def cast_array_c2d(self,arr_in, arr_out):
        arr_size = arr_in.size
        WGS, WGS_tot = self.get_wgs(arr_size)
        self._cast_array_d2c_knl(self.queue, (WGS_tot, ), (WGS, ),
                                 arr_in.data, arr_out.data,
                                 np.uint32(arr_size)).wait()

    def set_to(self, arr, val):
        if self.dev_type=='CPU' and arr.dtype == np.complex128:
            # just a workaround the stupid Apple CL implementation for CPU..
            arr_size = arr.size
            WGS, WGS_tot = self.get_wgs(arr_size)
            self._set_cdouble_to_knl(self.queue, (WGS_tot, ), (WGS, ),
                                     arr.data, np.complex128(val),
                                     np.uint32(arr_size)).wait()
        else:
            arr.fill(val)

    def append_c2c(self, arr_base, arr_add):
        arr_size = arr_base.size
        WGS, WGS_tot = self.get_wgs(arr_size)
        self._append_c2c_knl(self.queue, (WGS_tot, ), (WGS, ),
                             arr_base.data, arr_add.data,
                             np.uint32(arr_size)).wait()

    def mult_elementwise(self, x, z):
        arr_size = x.size
        WGS, WGS_tot = self.get_wgs(arr_size)
        self._mult_elementwise_knl(self.queue, (WGS_tot, ), (WGS, ),
            x.data, z.data, np.uint32(arr_size) ).wait()

    def axpbyz(self, a, x, b, y, z):
        arr_size = x.size
        WGS, WGS_tot = self.get_wgs(arr_size)
        self._axpbyz_c2c_knl(self.queue, (WGS_tot, ), (WGS, ),
                             np.complex128(a), x.data,
                             np.complex128(b), y.data,
                             z.data, np.uint32(arr_size) ).wait()

    def zpaxz(self, z, a, x):
        arr_size = x.size
        WGS, WGS_tot = self.get_wgs(arr_size)
        self._zpaxz_c2c_knl(self.queue, (WGS_tot, ), (WGS, ),
                            np.complex128(a), x.data,
                            z.data, np.uint32(arr_size) ).wait()


    def ab_dot_x(self, a, b, x, z):
        arr_size = x.size
        WGS, WGS_tot = self.get_wgs(arr_size)
        self._ab_dot_x_knl(self.queue, (WGS_tot, ), (WGS, ),
                           np.complex128(a), b.data, x.data,
                           z.data, np.uint32(self.Args['NxNrm1']),
                           np.uint32(self.Args['Nx']) ).wait()

    def import_comm(self, comm):
        self.comm = comm
        self.queue = comm.queue
        self.ctx = comm.ctx
        self.thr = comm.thr
        self.dev_type = comm.dev_type
        self.plat_name = comm.plat_name

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

        selected_dev = self.queue.device
        self.dev_type = device_type.to_string(selected_dev.type)
        self.dev_name = self.queue.device.name
        self.plat_name = selected_dev.platform.vendor
        self.ocl_version = selected_dev.opencl_c_version

        print("{} device {} is chosen on {} platform with {} compiler".
              format(self.dev_type, self.dev_name,
                     self.plat_name, self.ocl_version))

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


