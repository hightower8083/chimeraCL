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
                open(src_path + "particles_generic.cl").readlines())

        particles_sources = self.block_def_str + particles_sources

        prg = Program(self.ctx, particles_sources).\
            build(options=compiler_options)

        self._data_align_dbl_knl = prg.data_align_dbl
        self._data_align_int_knl = prg.data_align_int
        self._index_and_sum_knl = prg.index_and_sum_in_cell
        self._push_xyz_knl = prg.push_xyz
        self._push_p_boris_knl = prg.push_p_boris
        self._fill_grid_knl = prg.fill_grid

    def add_new_particles(self):
        args_strs = ['x', 'y', 'z', 'px', 'py', 'pz', 'w', 'g_inv']
        old_Np = self.DataDev['x'].size
        new_Np = self.DataDev['x_new'].size
        full_Np = old_Np + new_Np
        for arg in args_strs:
            buff = self.dev_arr(dtype=self.DataDev[arg].dtype,
                                shape=full_Np,
                                allocator=self.DataDev[arg + '_mp'])
            buff[:old_Np] = self.DataDev[arg]
            buff[old_Np:] = self.DataDev[arg+'_new']
            self.DataDev[arg] = buff
            self.DataDev[arg+'_new'] = None

        self.reset_num_parts()
        self.realloc_field_arrays()

    def realloc_field_arrays(self):
        flds_comps_str = []
        for fld_str in ('E', 'B'):
            for comp_str in ('x', 'y', 'z'):
                flds_comps_str.append(fld_str + comp_str)

        for arg in flds_comps_str:
            self.DataDev[arg] = self.dev_arr(val=0, dtype=np.double,
                shape=self.Args['Np'], allocator=self.DataDev[arg + '_mp'])

    def make_new_domain(self, parts_in):
        args_strs =  ['x', 'y', 'z', 'px', 'py', 'pz', 'w']

        xmin, xmax, rmin, rmax = \
          [parts_in[arg] for arg in ['Xmin', 'Xmax', 'Rmin', 'Rmax']]
        Nx_loc = int( np.ceil((xmax-xmin) / self.Args['dx']) + 1)
        Nr_loc = int( np.round((rmax-rmin) / self.Args['dr']) + 1)
        Xgrid_loc = self.dev_arr(val=(xmin+self.Args['dx']*np.arange(Nx_loc)),
                                 allocator=self.DataDev['Xgrid_loc_mp'])
        Rgrid_loc = self.dev_arr(val=(rmin+self.Args['dr']*np.arange(Nr_loc)),
                                 allocator=self.DataDev['Rgrid_loc_mp'])

        self.Args['right_lim'] = Xgrid_loc[-1].get()

        Ncells_loc = (Nx_loc-1)*(Nr_loc-1)
        Np = int(Ncells_loc*np.prod(self.Args['Nppc']))

        for arg in args_strs:
            self.DataDev[arg+'_new'] = self.dev_arr(shape=Np,
                dtype=np.double, allocator=self.DataDev[arg + '_new_mp'])

        theta_variator = self.dev_arr(shape=Ncells_loc,
            dtype=np.double, allocator=self.DataDev['theta_variator_mp'])
        self._fill_arr_rand(theta_variator, xmin=0, xmax=2*np.pi)

        gn_strs = ['x', 'y', 'z', 'w']
        gn_args = [self.DataDev[arg+'_new'].data for arg in gn_strs]
        gn_args += [theta_variator.data, ]
        gn_args += [Xgrid_loc.data, Rgrid_loc.data,
                    np.uint32(Nx_loc), np.uint32(Ncells_loc)]
        gn_args += list(np.array(self.Args['Nppc'], dtype=np.uint32))

        WGS, WGS_tot = self.get_wgs(Ncells_loc)
        self._fill_grid_knl(self.queue, (WGS_tot, ), (WGS, ), *gn_args).wait()

        self.DataDev['w_new'] *= self.Args['w0']

        for arg in ['px', 'py', 'pz']:
            if ('d'+arg in parts_in):
                self._fill_arr_randn(self.DataDev[arg+'_new'],
                                    mu=parts_in[arg+'_c'],
                                    sigma=parts_in['d'+arg])
            else:
                if arg+'_c' not in parts_in:
                    parts_in[arg+'_c'] = 0
                self.DataDev[arg+'_new'].fill(parts_in[arg+'_c'])

        if (parts_in['px_c'] != 0) and (parts_in['dpx'] != 0) and \
          (parts_in['py_c'] != 0) and (parts_in['dpy'] != 0) and \
          (parts_in['pz_c'] != 0) and (parts_in['dpz'] != 0):

            self.DataDev['g_inv_new'] = 1./sqrt(
                1 + self.DataDev['px_new']*self.DataDev['px_new']
                + self.DataDev['py_new']*self.DataDev['py_new']
                + self.DataDev['pz_new']*self.DataDev['pz_new'])
        else:
            self.DataDev['g_inv_new'] = self.dev_arr(shape=Np,val=1.0,
                dtype=np.double, allocator=self.DataDev['g_inv_new_mp'])

    def make_new_beam(self, parts_in):
        Np = parts_in['Np']

        args_strs =  ['x', 'y', 'z', 'px', 'py', 'pz', 'w']
        for arg in args_strs:
            self.DataDev[arg+'_new'] = self.dev_arr(
                shape=Np, dtype=np.double)

        for arg in ['x', 'y', 'z']:
            self._fill_arr_randn(self.DataDev[arg+'_new'],
                                mu=parts_in[arg+'_c'],
                                sigma=parts_in['L'+arg])

        for arg in ['px', 'py', 'pz']:
            if arg+'_c' not in parts_in:
                parts_in[arg+'_c'] = 0
            if 'd'+arg not in parts_in:
                parts_in['d'+arg] = 0

            self._fill_arr_randn(self.DataDev[arg+'_new'],
                                mu=parts_in[arg+'_c'],
                                sigma=parts_in['d'+arg])

        self.DataDev['w_new'][:] = parts_in['FullCharge']/parts_in['Np']

        self.DataDev['g_inv_new'] = 1./sqrt(
            1 + self.DataDev['px_new']*self.DataDev['px_new']
            + self.DataDev['py_new']*self.DataDev['py_new']
            + self.DataDev['pz_new']*self.DataDev['pz_new'])

    def reset_num_parts(self):
        Np = self.DataDev['x'].size
        self.DataDev['Np'].fill(Np)
        self.Args['Np'] = Np

    def push_coords(self, mode='half'):
        if self.Args['Np']==0:
            return

        WGS, WGS_tot = self.get_wgs(self.Args['Np'])

        if mode=='half':
            which_dt = 'dt_2'
        else:
            which_dt = 'dt'

        args_strs =  ['x', 'y', 'z', 'px', 'py', 'pz', 'g_inv', which_dt, 'Np']
        args = [self.DataDev[arg].data for arg in args_strs]
        self._push_xyz_knl(self.queue, (WGS_tot, ), (WGS, ), *args).wait()

    def push_veloc(self):
        if self.Args['Np'] == 0:
            return

        WGS, WGS_tot = self.get_wgs(self.Args['Np'])

        args_strs =  ['px','py','pz','g_inv',
                      'Ex','Ey','Ez',
                      'Bx','By','Bz','FactorPush','Np']

        args = [self.DataDev[arg].data for arg in args_strs]
        self._push_p_boris_knl(self.queue, (WGS_tot, ), (WGS, ), *args).wait()

    def index_sort(self, grid):
        WGS, WGS_tot = self.get_wgs(self.Args['Np'])

        self.DataDev['indx_in_cell'] = self.dev_arr(dtype=np.uint32,
            shape=self.Args['Np'], allocator=self.DataDev['indx_in_cell_mp'])

        self.DataDev['cell_offset'] = self.dev_arr(val=0,
            dtype=np.uint32, shape=grid.Args['Nxm1Nrm1'],
            allocator=self.DataDev['cell_offset_mp'])

        part_strs =  ['x', 'y', 'z', 'cell_offset', 'Np']
        grid_strs =  ['Nx', 'Xmin', 'dx_inv',
                      'Nr', 'Rmin', 'dr_inv']

        args = [self.DataDev[arg].data for arg in part_strs] + \
               [self.DataDev['indx_in_cell'].data, ] + \
               [grid.DataDev[arg].data for arg in grid_strs]

        self._index_and_sum_knl(self.queue, (WGS_tot, ), (WGS, ), *args).wait()
        self.DataDev['cell_offset'] = self._cumsum(self.DataDev['cell_offset'])

        if self.comm.sort_method == 'Radix':
            indx_size = self.DataDev['indx_in_cell'].size
            self.DataDev['sort_indx'] = arange(self.queue, 0, indx_size, 1,
                dtype=np.uint32, allocator=self.DataDev['sort_indx_mp'])

            (self.DataDev['indx_in_cell'], self.DataDev['sort_indx']), evnt = \
                self._sort_rdx_knl(self.DataDev['indx_in_cell'],
                self.DataDev['sort_indx'])

            evnt.wait()
        elif self.comm.sort_method == 'NumPy':
                self.DataDev['sort_indx'] = \
                    self.DataDev['indx_in_cell'].get().argsort()

                self.DataDev['sort_indx'] = to_device(
                    self.queue, self.DataDev['sort_indx'])

    def align_and_damp(self, comps_align):
        num_staying = self.DataDev['cell_offset'][-1].get().item()

        if num_staying == 0:
            for comp in comps_align + ['sort_indx',]:
                self.DataDev[comp] = self.dev_arr(shape=0,
                    dtype=self.DataDev[comp].dtype,
                    allocator=self.DataDev[comp + '_mp'])
            self.reset_num_parts()
            return

        WGS, WGS_tot = self.get_wgs(num_staying)
        for comp in comps_align:
            buff_parts = self.dev_arr(dtype=self.DataDev[comp].dtype,
                                      shape=(num_staying, ),
                                      allocator=self.DataDev[comp + '_mp'])

            self._data_align_dbl_knl(self.queue, (WGS_tot, ), (WGS, ),
                                     self.DataDev[comp].data,
                                     buff_parts.data,
                                     self.DataDev['sort_indx'].data,
                                     np.uint32(num_staying)).wait()
            self.DataDev[comp] = buff_parts

        self.DataDev['sort_indx'] = arange(self.queue, 0, num_staying, 1,
            dtype=np.uint32, allocator=self.DataDev['sort_indx_mp'])
        self.reset_num_parts()

    def _fill_arr_randn(self, arr, mu=0, sigma=1):
        return self._generator_knl.fill_normal(ary=arr, queue=self.queue,
                                               mu=mu, sigma=sigma)

    def _fill_arr_rand(self, arr, xmin=0, xmax=1):
        return self._generator_knl.fill_uniform(ary=arr, queue=self.queue,
                                                a=xmin,b=xmax)

    def _cumsum(self,arr_in):
        evnt, arr_tmp = cumsum(arr_in, return_event=True, queue=self.queue)
        arr_out = self.dev_arr(dtype=np.uint32, shape=arr_tmp.size+1,
                               allocator=self.DataDev['sort_indx_mp'])
        arr_out[0] = 0
        evnt.wait()
        arr_out[1:] = arr_tmp[:]
        return arr_out
