from numpy import double
from numpy import ndarray
from numpy import ceil

from pyopencl.array import zeros
from pyopencl.array import empty
from pyopencl import CommandQueue
from pyopencl import device_type
from pyopencl import create_some_context

from reikna.cluda import ocl_api

compiler_options = ['-cl-fast-relaxed-math',]
#compiler_options = []

class GenericMethodsCL:
    def dev_arr(self, val=None, shape=(1,), dtype=double):
        if type(val) is ndarray:
            arr = self.thr.to_device(val)
        elif val==0:
            arr = zeros(self.queue, shape, dtype=dtype)
        else:
            arr = empty(self.queue, shape, dtype=dtype)
            if val is not None: arr.fill(val)
        return arr

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
            WGS_tot = int(ceil(1.*Nelem/self.WGS))*self.WGS
            WGS = self.WGS
            return WGS, WGS_tot

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


