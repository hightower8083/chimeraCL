from pyopencl.clrandom import ThreefryGenerator
from pyopencl.algorithm import RadixSort
from pyopencl.array import arange, cumsum
from pyopencl import enqueue_marker, enqueue_barrier
from pyopencl import Program

from numpy import uint32, ceil

from .generic_methods_cl import GenericMethodsCL
from .generic_methods_cl import compiler_options

class ParticleMethodsCL(GenericMethodsCL):
    def _compile_methods(self):
        self.set_global_working_group_size()

        self._generator_knl = ThreefryGenerator(context=self.ctx)

        self._sort_rdx_knl = RadixSort(self.ctx,
                    "uint *indx_in_cell, uint *indx_init",
                    key_expr="indx_in_cell[i]", index_dtype=uint32,
                    sort_arg_names=["indx_in_cell","indx_init"],
                    options=compiler_options)

        particles_sources = ''.join( open("./kernels/particles_generic.cl")\
                                     .readlines())

        particles_sources = self.block_def_str + particles_sources

        prg = Program(self.ctx, particles_sources).\
            build(options=compiler_options)

        self._data_align_dbl_knl = prg.data_align_dbl
        self._data_align_dbl_tst_knl = prg.data_align_dbl_tst
        self._data_align_int_knl = prg.data_align_int
        self._index_and_sum_knl = prg.index_and_sum_in_cell
        self._index_compare_sort_knl = prg.index_compare_sort
        self._push_xyz_knl = prg.push_xyz
        self._push_p_boris_knl = prg.push_p_boris

    def push_coords(self,mode='half'):
        WGS, WGS_tot = self.get_wgs(self.Args['Np'])

        if mode=='half':
            args_strs =  ['x','y','z', 'px','py','pz','g_inv','dt_2','Np']
        else:
            args_strs =  ['x','y','z', 'px','py','pz','g_inv','dt','Np']

        args = [self.DataDev[arg].data for arg in args_strs]
        self._push_xyz_knl(self.queue,
                           (WGS_tot,),(WGS,),
                           *args).wait()

    def push_veloc(self):
        WGS, WGS_tot = self.get_wgs(self.Args['Np'])

        args_strs =  ['px','py','pz','g_inv',
                      'Ex','Ey','Ez',
                      'Bx','By','Bz','dt','Np']

        args = [self.DataDev[arg].data for arg in args_strs]
        self._push_p_boris_knl(self.queue,
                               (WGS_tot,),(WGS,),
                               *args).wait()

    def index_and_sum(self):
        WGS_tot = int(ceil(self.Args['Np']*1./self.WGS))*self.WGS

        WGS, WGS_tot = self.get_wgs(self.Args['Np'])
        for arg in ['sum_in_cell','indx_in_cell','cell_offset']:
            self.DataDev[arg][:] = 0

        args_strs =  ['x','y','z','indx_in_cell', 'sum_in_cell',
                      'Nx','Xmin','dx_inv',
                      'Nr','Rmin','dr_inv', 'Np']

        args = [self.DataDev[arg].data for arg in args_strs]
        self._index_and_sum_knl(self.queue,
                                (WGS_tot,),(WGS,),
                                *args).wait()
        self.DataDev['cell_offset'] = self._cumsum(self.DataDev['sum_in_cell'])

    def sort_rdx(self, indx):
        sorting_indx = arange(self.queue,0,indx.size,1,dtype=uint32)
        [indx,sorting_indx], evnt = self._sort_rdx_knl(indx,sorting_indx)
        evnt.wait()
        return sorting_indx

    def align_and_damp(self, idx, comps):
        num_staying = self.DataDev['cell_offset'][-1].get().item()
        size_cl = self.dev_arr(val = num_staying,dtype=uint32)

        WGS, WGS_tot = self.get_wgs(num_staying)

        for comp in comps:
            buff = self.dev_arr(dtype=self.DataDev[comp].dtype,
                                shape=(num_staying,))
            self._data_align_dbl_knl(self.queue,
                         (WGS_tot,), (WGS,),
                         self.DataDev[comp].data, buff.data,
                         idx.data, uint32(num_staying)).wait()

            self.DataDev[comp] = buff

        self.DataDev['indx_in_cell'] = self.DataDev['indx_in_cell']\
                                                        [:num_staying]
        self.reset_num_parts()

    def reset_num_parts(self):
        Np = self.DataDev['x'].size
        self.DataDev['Np'].fill(Np)
        self.Args['Np'] = Np

    def fill_arr_randn(self, arr, mu=0, sigma=1):
        self._generator_knl.fill_normal(ary=arr, queue=self.queue,
                                   mu=mu, sigma=sigma)
        enqueue_barrier(self.queue)

    def _cumsum(self,arr_in):
        evnt, arr_tmp = cumsum(arr_in, return_event=True, queue=self.queue)
        evnt.wait()
        arr_out = self.dev_arr(val=0, dtype=uint32, shape=arr_tmp.size+1)
        arr_out[1:] = arr_tmp[:]
        return arr_out

#############################################################
##############################  TESTING  ####################
#############################################################

    def sort_on_grid(self):
        self.DataDev['sum_in_cell'][:] = 0

        args_strs =  ['indx_in_cell','cell_offset','sum_in_cell',
                      'Nxm1Nrm1','Np']
        args = [self.DataDev[arg].data for arg in args_strs]

        out_sum = self.dev_arr(dtype=uint32,val=0)
        sorting_indx = self.dev_arr(val=0,dtype=uint32, shape=self.Args['Np'])

        self._index_compare_sort_knl(self.queue,
                                     (self.Args['Np'],), None,
                                     sorting_indx.data, out_sum.data,
                                     *args).wait()

        buff = self.dev_arr(val=0,dtype=self.DataDev['indx_in_cell'].dtype,
                            shape=self.Args['Np'])

        self._data_align_int_knl(self.queue,
                                (self.Args['Np'],), None,
                                 self.DataDev['indx_in_cell'].data,
                                 buff.data, sorting_indx.data,
                                 self.DataDev['Np'].data).wait()

        self.DataDev['indx_in_cell'][:] = buff

        self.DataDev['cell_offset'] = self._cumsum(self.DataDev['sum_in_cell'])
        return sorting_indx

    def align_and_damp_tst(self, idx, comps):
        num_staying = self.DataDev['cell_offset'][-1].get().item()
        size_cl = self.dev_arr(val = num_staying,dtype=uint32)

        WGS, WGS_tot = self.get_wgs(num_staying)

        for comp in comps:
            buff = self.dev_arr(dtype=self.DataDev[comp].dtype,
                                shape=(num_staying,))
            self._data_align_dbl_tst_knl(self.queue,
                         (WGS_tot,), (WGS,),
                         self.DataDev[comp].data, buff.data,
                         idx.data, uint32(num_staying)).wait()

            self.DataDev[comp] = buff

        self.DataDev['indx_in_cell'] = self.DataDev['indx_in_cell']\
                                                        [:num_staying]
        self.reset_num_parts()

