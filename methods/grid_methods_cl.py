from numpy import uint32, double,complex128
from numpy import ceil, arange

from pyopencl import Program
from pyopencl import enqueue_marker, enqueue_barrier

from pyopencl.array  import to_device, empty_like

from reikna.fft import FFT
from reikna.linalg import MatrixMul

from .generic_methods_cl import GenericMethodsCL
from .generic_methods_cl import compiler_options

from chimeraCL import __path__ as src_path

src_path = src_path[0] + '/kernels/'

class GridMethodsCL(GenericMethodsCL):
    def _compile_methods(self):
        self.set_global_working_group_size()

        grid_sources = []
        grid_sources.append( ''.join(open(src_path+"grid_generic.cl")\
              .readlines()) )

        grid_sources.append( ''.join( \
              open(src_path+"grid_deposit_m"+str(self.Args['M'])+".cl")\
              .readlines() ) )

        grid_sources = self.block_def_str + ''.join(grid_sources)

        prg = Program(self.ctx, grid_sources).\
            build(options=compiler_options)

        self._divide_by_r_dbl_knl = prg.divide_by_r_dbl
        self._divide_by_r_clx_knl = prg.divide_by_r_clx
        self._treat_axis_dbl_knl = prg.treat_axis_dbl
        self._treat_axis_clx_knl = prg.treat_axis_clx
        self._set_cdouble_to_zero_knl = prg.set_cdouble_to_zero
        self._cast_array_d2c_knl = prg.cast_array_d2c

        self._depose_scalar_knl = prg.depose_scalar
        self._depose_vector_knl = prg.depose_vector
        self._project_scalar_knl = prg.project_scalar
        self._project_vec6_knl = prg.project_vec6

        init_array_c = self.DataDev['Ex_fb_m0'].copy()

        self._prepare_fft(init_array_c)

        init_array_c = self.DataDev['Ex_fb_m0'].copy()
        init_array_d = init_array_c.real

        self._prepare_dot(init_array_d,init_array_c)

    def depose_scalar(self, parts, sclr, fld):
        # Depose weights by 4-cell-grid scheme
        WGS, WGS_tot = self.get_wgs(self.Args['NxNr_4'])

        part_strs = ['sort_indx','x','y','z',
                     sclr,'cell_offset',]
        grid_strs = ['Nx', 'Xmin', 'dx_inv',
                     'Nr', 'Rmin', 'dr_inv',
                     'NxNr_4']
        fld_strs = [fld+'_m'+str(m) for m in range(self.Args['M']+1)]

        args_parts = [parts.DataDev[arg].data for arg in part_strs]
        args_grid = [self.DataDev[arg].data for arg in grid_strs]
        args_fld = [self.DataDev[arg].data for arg in fld_strs]

        args = args_parts + args_grid + args_fld

        evnt = enqueue_marker(self.queue)
        for i_off in arange(4).astype(uint32):
            evnt = self._depose_scalar_knl(self.queue,
                                           (WGS_tot,),(WGS,),
                                           i_off, *args,
                                           wait_for = [evnt,])

        # Correct near axis deposition
        args_grid = [self.DataDev[fld+'_m'+str(m)].data \
                     for m in range(self.Args['M']+1)]

        WGS, WGS_tot = self.get_wgs(self.Args['Nx'])
        enqueue_barrier(self.queue)
        self._treat_axis_dbl_knl(self.queue,
                                 (WGS_tot,), (WGS,),
                                 args_grid[0], uint32(self.Args['Nx']))

        for m in range(1,self.Args['M']+1):
            self._treat_axis_clx_knl(self.queue,
                                     (WGS_tot,), (WGS,),
                                     args_grid[m], uint32(self.Args['Nx']))
        # Divide by radius
        WGS, WGS_tot = self.get_wgs(self.Args['NxNr'])
        grid_strs =  ['NxNr','Nx','Rgrid_inv']
        grid_args = [self.DataDev[arg].data for arg in grid_strs]

        enqueue_barrier(self.queue)
        self._divide_by_r_dbl_knl(self.queue,(WGS_tot,), (WGS,),
                             args_grid[0],*grid_args)

        for m in range(1,self.Args['M']+1):
            self._divide_by_r_clx_knl(self.queue,(WGS_tot,), (WGS,),
                                 args_grid[m],*grid_args)
        enqueue_barrier(self.queue)


    def project_scalar(self, parts, sclr, fld):
        # Project a scalar by 4-cell-grid scheme:
        #   to be replaced with normal scheme
        WGS, WGS_tot = self.get_wgs(self.Args['Nxm1Nrm1'])

        part_strs = ['sort_indx',sclr,'x','y','z','cell_offset']

        grid_strs = ['Nx', 'Xmin', 'dx_inv',
                     'Nr', 'Rmin', 'dr_inv',
                     'Nxm1Nrm1']

        fld_strs = [fld+'_m'+str(m) for m in range(self.Args['M']+1)]

        args_parts = [parts.DataDev[arg].data for arg in part_strs]
        args_grid = [self.DataDev[arg].data for arg in grid_strs]
        args_fld = [self.DataDev[arg].data for arg in fld_strs]
        
        args = args_parts + args_grid + args_fld
        evnt = self._project_scalar_knl(self.queue, (WGS_tot,),(WGS,),*args)

    def depose_vector(self, parts, vec, factors,vec_fld):
        # Depose weights by 4-cell-grid scheme

        part_strs =  ['sort_indx','x','y','z'] + vec + factors + \
                     ['cell_offset']

        grid_strs = ['Nx', 'Xmin', 'dx_inv',
                     'Nr', 'Rmin', 'dr_inv',
                     'NxNr_4']
        fld_strs = []

        for m in range(self.Args['M']+1):
            for comp in ('x','y','z'):
                fld_strs.append(vec_fld+comp+'_m'+str(m))

        args_parts = [parts.DataDev[arg].data for arg in part_strs]
        args_grid = [self.DataDev[arg].data for arg in grid_strs]
        args_fld = [self.DataDev[arg].data for arg in fld_strs]

        args_dep = args_parts + args_grid + args_fld

        args_raddiv_strs =  ['NxNr','Nx','Rgrid_inv']
        args_raddiv = [self.DataDev[arg].data for arg in args_raddiv_strs]

        WGS, WGS_tot = self.get_wgs(self.Args['NxNr_4'])
        evnt = enqueue_marker(self.queue)
        for i_off in arange(4).astype(uint32):
            evnt = self._depose_vector_knl(self.queue,
                                          (WGS_tot,),(WGS,),
                                          i_off, *args_dep,
                                          wait_for = [evnt,])
        enqueue_barrier(self.queue)

        for fld in [vec_fld + comp for comp in ('x','y','z')]:
            # Correct near axis deposition
            args_fld = [self.DataDev[fld+'_m'+str(m)].data \
                         for m in range(self.Args['M']+1)]

            WGS, WGS_tot = self.get_wgs(self.Args['Nx'])
            self._treat_axis_dbl_knl(self.queue,
                                     (WGS_tot,), (WGS,),
                                     args_fld[0], uint32(self.Args['Nx']))

            for m in range(1,self.Args['M']+1):
                self._treat_axis_clx_knl(self.queue,
                                         (WGS_tot,), (WGS,),
                                         args_fld[m], uint32(self.Args['Nx']))

            # Divide by radius
            WGS, WGS_tot = self.get_wgs(self.Args['NxNr'])
            enqueue_barrier(self.queue)
            self._divide_by_r_dbl_knl(self.queue,(WGS_tot,), (WGS,),
                                 args_fld[0],*args_raddiv)

            for m in range(1,self.Args['M']+1):
                self._divide_by_r_clx_knl(self.queue,(WGS_tot,), (WGS,),
                                     args_fld[m],*args_raddiv)
            enqueue_barrier(self.queue)

    def project_vec6(self, parts, vecs, flds):
        # Project 2 fields by 4-cell-grid scheme
        #   to be replaced with normal scheme
        WGS, WGS_tot = self.get_wgs(self.Args['Nxm1Nrm1'])


        parts_strs = ['sort_indx',] + \
                     [vecs[0] + comp for comp in ('x','y','z')] + \
                     [vecs[1] + comp for comp in ('x','y','z')] + \
                     ['x','y','z','cell_offset',]

        grid_strs = ['Nx', 'Xmin', 'dx_inv',
                     'Nr', 'Rmin', 'dr_inv',
                     'Nxm1Nrm1']

        flds_strs = [flds[0] + comp for comp in ('x','y','z')] + \
                    [flds[1] + comp for comp in ('x','y','z')]

        args_parts = [parts.DataDev[arg].data for arg in parts_strs]
        args_grid = [self.DataDev[arg].data for arg in grid_strs]

        args_fld = []
        for fld in flds_strs:
            args_fld += [self.DataDev[fld+'_m'+str(m)].data \
                         for m in range(self.Args['M']+1)]

        args = args_parts + args_grid + args_fld
        evnt = self._project_vec6_knl(self.queue, (WGS_tot,),(WGS,),*args)

    def transform_field(self, arg,dir):
        if dir == 0:
            dht_arg = 'DHT_m'
            arg_in = arg + '_m'
            arg_out = arg + '_fb_m'

            self.DataDev['field_fb_aux1_dbl'] = self.DataDev[arg_in+'0']\
                                                  [1:].copy()
                                                  
            self.DataDev['field_fb_aux2_dbl'] = \
                self._dot0(self.DataDev[dht_arg+'0'],
                           self.DataDev['field_fb_aux1_dbl'])

            enqueue_barrier(self.queue)

            self.DataDev[arg_out+'0'][:] = self.DataDev['field_fb_aux2_dbl']\
                                                   .astype(complex128)
            enqueue_barrier(self.queue)
            self.DataDev[arg_out+'0'] = self._fft(self.DataDev[arg_out+'0'],
                                                  dir=dir)

            for m in range(1,self.Args['M']+1):
                arg_in_m = arg_in + str(m)
                arg_out_m = arg_out + str(m)


                self.DataDev['field_fb_aux1_clx'] = self.DataDev[arg_in_m]\
                                                      [1:,:].copy()

                self.DataDev[arg_out_m] = \
                    self._dot1(self.DataDev[dht_arg+str(m)],
                               self.DataDev['field_fb_aux1_clx'])
                enqueue_barrier(self.queue)
                self.DataDev[arg_out_m] = self._fft(self.DataDev[arg_out_m],
                                                    dir=dir)
        elif dir == 1:
            dht_arg = 'DHT_inv_m'
            arg_in = arg + '_fb_m'
            arg_out = arg + '_m'

            self.DataDev['field_fb_aux1_clx'] = \
                self._fft(self.DataDev[arg_in+'0'], dir=dir)

            self._cast_array_c2d(self.DataDev['field_fb_aux1_clx'],
                                self.DataDev['field_fb_aux1_dbl'])

            self.DataDev['field_fb_aux2_dbl'] = \
                self._dot0(self.DataDev[dht_arg+'0'],
                           self.DataDev['field_fb_aux1_dbl'])

            enqueue_barrier(self.queue)
            self.DataDev[arg_out+'0'][1:] = self.DataDev['field_fb_aux2_dbl']

            for m in range(1,self.Args['M']+1):

                arg_in_m = arg_in + str(m)
                arg_out_m = arg_out + str(m)
                self.DataDev['field_fb_aux1_clx'] = \
                    self._fft(self.DataDev[arg_in_m], dir=dir)

                self.DataDev['field_fb_aux2_clx'] = \
                    self._dot1(self.DataDev[dht_arg+str(m)],
                               self.DataDev['field_fb_aux1_clx'])

                enqueue_barrier(self.queue)
                self.DataDev[arg_out_m][1:] = self.DataDev['field_fb_aux2_clx']

    def set_to_zero(self, arr):
        if arr.dtype != complex128:
            arr.fill(0)
        else:
            arr_size = arr.size
            WGS, WGS_tot = self.get_wgs(arr_size)
            self._set_cdouble_to_zero_knl(self.queue, (WGS_tot, ), (WGS, ),
                                          arr.data, uint32(arr_size)).wait()

    def _cast_array_c2d(self,arr_in, arr_out):
        arr_size = arr_in.size
        WGS, WGS_tot = self.get_wgs(arr_size)
        self._cast_array_d2c_knl(self.queue,(WGS_tot,),(WGS,),
                                 arr_in.data, arr_out.data,
                                 uint32(arr_size)).wait()

    def _prepare_dot(self, init_array_d,init_array_c):
        if self.comm.dot_method=='Reikna':
            self._dot0_knl = MatrixMul(self.DataDev['DHT_m0'],init_array_d, \
                                      out_arr=init_array_d)\
                                       .compile(self.thr,fast_math=True)
            self._dot_knl = MatrixMul(self.DataDev['DHT_m0'], init_array_c, \
                                      out_arr=init_array_c)\
                                      .compile(self.thr,fast_math=True)
            def dot0_wrp(a,b):
                c = empty_like(b)
                self._dot0_knl(c,a,b)
                return c

            def dot1_wrp(a,b):
                c = empty_like(b)
                self._dot1_knl(c,a,b)
                return c

            self._dot0 = dot0_wrp
            self._dot1  = dot1_wrp

        elif self.comm.dot_method=='NumPy':
            from numpy import dot as dot_np

            def dot_wrp(a,b):
                c = dot_np(a.get(),b.get())
                c = to_device(self.queue,c)
                return c

            self._dot0 = dot_wrp
            self._dot1  = dot_wrp

    def _prepare_fft(self,input_array):
        input_array = self.DataDev['Ex_fb_m0'].copy()
        if self.comm.fft_method=='pyFFTW':
            from pyfftw import empty_aligned,FFTW

            self.arr_fft_in = empty_aligned(input_array.shape,
                        dtype=complex128, n=16)
            self.arr_fft_out = empty_aligned(input_array.shape,
                        dtype=complex128, n=16)
            self._fft_knl = [FFTW(input_array=self.arr_fft_in,
                              output_array=self.arr_fft_out,
                              direction = 'FFTW_FORWARD', threads=4,axes=(1,)),
                         FFTW(input_array=self.arr_fft_in,
                              output_array=self.arr_fft_out,
                              direction = 'FFTW_BACKWARD', threads=4,axes=(1,))]

        elif self.comm.fft_method=='Reikna':
            fft = FFT(input_array,axes=(1,))
            self._fft_knl = fft.compile(self.thr)

    def _fft(self,arr, dir=0):
        if self.comm.fft_method=='pyFFTW':
            self.arr_fft_in[:] = arr.get()
            self._fft_knl[dir]()
            arr_out = self.arr_fft_out
        else:
            arr_out = empty_like(arr)
            self._fft_knl(arr_out,arr,dir)

        return arr_out
