import numpy as np
from pyopencl.clrandom import ThreefryGenerator
from pyopencl.algorithm import RadixSort
from pyopencl.array import arange, cumsum, to_device
from pyopencl import enqueue_marker, enqueue_barrier
from pyopencl import Program
from pyopencl.clmath import sqrt as sqrt

from .generic_methods_cl import GenericMethodsCL
from .generic_methods_cl import compiler_options
from chimeraCL import __path__ as src_path

src_path = src_path[0] + '/kernels/'


class ParticleMethodsCL(GenericMethodsCL):
    def init_particle_methods(self):
        self.init_generic_methods()
        self.set_global_working_group_size()

        self._generator_knl = ThreefryGenerator(context=self.ctx)

        self._sort_rdx_knl = RadixSort(self.ctx,
                    "uint *indx_in_cell, uint *indx_init",
                    key_expr="indx_in_cell[i]", index_dtype=np.uint32,
                    sort_arg_names=["indx_in_cell", "indx_init"],
                    options=compiler_options)

        particles_sources = ''.join(
                open(src_path+"particles_generic.cl").readlines())

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

    def push_coords(self, mode='half'):
        WGS, WGS_tot = self.get_wgs(self.Args['Np'])

        if mode=='half':
            args_strs =  ['x','y','z', 'px','py','pz','g_inv','dt_2','Np']
        else:
            args_strs =  ['x','y','z', 'px','py','pz','g_inv','dt','Np']

        args = [self.DataDev[arg].data for arg in args_strs]
        self._push_xyz_knl(self.queue, (WGS_tot, ), (WGS, ), *args).wait()

    def push_veloc(self):
        WGS, WGS_tot = self.get_wgs(self.Args['Np'])

        args_strs =  ['px','py','pz','g_inv',
                      'Ex','Ey','Ez',
                      'Bx','By','Bz','dt','Np']

        args = [self.DataDev[arg].data for arg in args_strs]
        self._push_p_boris_knl(self.queue, (WGS_tot, ), (WGS, ), *args).wait()

    def index_and_sum(self, grid):
        WGS, WGS_tot = self.get_wgs(self.Args['Np'])

        self.DataDev['cell_offset'] = self.dev_arr(val=0, dtype=np.uint32,
                                                   shape=grid.Args['Nxm1Nrm1'])

        part_strs =  ['x','y','z','indx_in_cell',
                      'cell_offset','Np']

        grid_strs =  ['Nx','Xmin','dx_inv',
                      'Nr','Rmin','dr_inv']

        args = [self.DataDev[arg].data for arg in part_strs] + \
                    [grid.DataDev[arg].data for arg in grid_strs]
        self._index_and_sum_knl(self.queue, (WGS_tot, ), (WGS, ), *args).wait()
        self.DataDev['cell_offset'] = self._cumsum(self.DataDev['cell_offset'])

    def index_sort(self):
        if self.comm.sort_method == 'Radix':
            indx_size = self.DataDev['indx_in_cell'].size
            self.DataDev['sort_indx'] = arange(self.queue, 0, indx_size, 1,
                                               dtype=np.uint32)

            [indx, self.DataDev['sort_indx']], evnt = self._sort_rdx_knl(
                self.DataDev['indx_in_cell'], self.DataDev['sort_indx'])

            evnt.wait()
        elif self.comm.sort_method == 'NumPy':
                self.DataDev['sort_indx'] = self.DataDev['indx_in_cell'].\
                    get().argsort()
                self.DataDev['sort_indx'] = to_device(
                    self.queue, self.DataDev['sort_indx'])

    def align_and_damp(self, comps_align, comps_simple_dump):
        num_staying = self.DataDev['cell_offset'][-1].get().item()

        WGS, WGS_tot = self.get_wgs(num_staying)
        for comp in comps_align:
            buff = self.dev_arr(dtype=self.DataDev[comp].dtype,
                                shape=(num_staying,))
            self._data_align_dbl_knl(self.queue, (WGS_tot, ), (WGS, ),
                                     self.DataDev[comp].data, buff.data,
                                     self.DataDev['sort_indx'].data,
                                     np.uint32(num_staying)).wait()

            self.DataDev[comp] = buff

        for comp in comps_simple_dump:
            self.DataDev[comp] = self.DataDev[comp][:num_staying]

        self.reset_num_parts()
        self.DataDev['sort_indx'] = arange(self.queue, 0, self.Args['Np'], 1,
                                           dtype=np.uint32)

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
        arr_out = self.dev_arr(val=0, dtype=np.uint32, shape=arr_tmp.size+1)
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

        out_sum = self.dev_arr(dtype=np.uint32,val=0)
        sorting_indx = self.dev_arr(val=0,dtype=np.uint32, shape=self.Args['Np'])

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
        size_cl = self.dev_arr(val = num_staying,dtype=np.uint32)

        WGS, WGS_tot = self.get_wgs(num_staying)

        for comp in comps:
            buff = self.dev_arr(dtype=self.DataDev[comp].dtype,
                                shape=(num_staying,))
            self._data_align_dbl_tst_knl(self.queue,
                         (WGS_tot,), (WGS,),
                         self.DataDev[comp].data, buff.data,
                         idx.data, np.uint32(num_staying)).wait()

            self.DataDev[comp] = buff

        self.DataDev['indx_in_cell'] = self.DataDev['indx_in_cell']\
                                                        [:num_staying]
        self.reset_num_parts()

