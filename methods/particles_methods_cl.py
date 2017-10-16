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
        self._data_align_int_knl = prg.data_align_int
        self._index_and_sum_knl = prg.index_and_sum_in_cell
        self._push_xyz_knl = prg.push_xyz
        self._push_p_boris_knl = prg.push_p_boris
        self._fill_grid_knl = prg.fill_grid

    def prepare_generator_data(self):
        x_cell_grid, r_cell_grid, theta_cell_grid = \
            np.mgrid[0:1:1./self.Args['Nppc'][0],
                     0:1:1./self.Args['Nppc'][1],
                     0:1:1./self.Args['Nppc'][2]]

        x_cell_grid += 0.5/self.Args['Nppc'][0]
        r_cell_grid += 0.5/self.Args['Nppc'][1]
        theta_cell_grid *= 2*np.pi

        self.DataDev['cell_xrt'] = \
            [to_device(self.queue,x_cell_grid.flatten()),
             to_device(self.queue,r_cell_grid.flatten()),
             to_device(self.queue,theta_cell_grid.flatten())]

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

            (self.DataDev['indx_in_cell'], self.DataDev['sort_indx']), evnt = \
              self._sort_rdx_knl(self.DataDev['indx_in_cell'],
                                 self.DataDev['sort_indx'], )
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

    def make_parts(self, parts_in):
        args_strs =  ['x', 'y', 'z', 'px', 'py', 'pz', 'w',
                      'Ex', 'Ey', 'Ez','Bx', 'By', 'Bz']

        if parts_in['Type']=='beam':
            Np = parts_in['Np']

            for arg in args_strs:
                self.DataDev[arg] = self.dev_arr(shape=Np, dtype=np.double)

            for arg in ['x', 'y', 'z']:
                self.fill_arr_randn(self.DataDev[arg],
                                    mu=parts_in[arg+'_c'],
                                    sigma=parts_in['L'+arg])

            self.DataDev['w'][:] = self.Args['q']
        elif parts_in['Type']=='domain':
            xmin, xmax, dx, rmin, rmax, dr = \
              [parts_in[arg] for arg in ['xmin', 'xmax', 'dx',
                                       'rmin', 'rmax', 'dr']]
            Nx_loc = int(np.ceil((xmax-xmin) / dx))+1
            Nr_loc = int(np.ceil((rmax-rmin) / dr))+1
            Xgrid_loc = self.dev_arr(val = xmin+dx*np.arange(Nx_loc))
            Rgrid_loc = self.dev_arr(val = rmin+dr*np.arange(Nr_loc))

            Ncells_loc = (Nx_loc-1)*(Nr_loc-1)
            Np = int(Ncells_loc*np.prod(self.Args['Nppc']))

            for arg in args_strs:
                self.DataDev[arg] = self.dev_arr(shape=Np, dtype=np.double)

            self.DataDev['theta_variator'] = self.dev_arr(shape=Ncells_loc,
                                                          dtype=np.double)
            self.fill_arr_rand(self.DataDev['theta_variator'],
                               xmin=0, xmax=2*np.pi)

            gnrtr_strs = ['x', 'y', 'z', 'w', 'theta_variator']
            gnrtr_args = [self.DataDev[arg].data for arg in gnrtr_strs]
            gnrtr_args += [Xgrid_loc.data, Rgrid_loc.data,
                           np.uint32(Nx_loc), np.uint32(Ncells_loc)]
            gnrtr_args += list(np.array(self.Args['Nppc'],dtype=np.uint32))

            WGS, WGS_tot = self.get_wgs(Ncells_loc)

            self._fill_grid_knl(self.queue, (WGS_tot, ),
                                (WGS, ), *gnrtr_args).wait()
        else:
            print("Specify species Type as beam or domain")


        for arg in ['px', 'py', 'pz']:
            self.fill_arr_randn(self.DataDev[arg],
                                mu=parts_in[arg+'_c'],
                                sigma=parts_in['d'+arg])

        self.DataDev['g_inv'] = 1./sqrt(
            1 + self.DataDev['px']*self.DataDev['px']
            + self.DataDev['py']*self.DataDev['py']
            + self.DataDev['pz']*self.DataDev['pz'])

        self.DataDev['indx_in_cell'] = self.dev_arr(dtype=np.uint32,
                                                    shape=Np)
        self.reset_num_parts()

    def reset_num_parts(self):
        Np = self.DataDev['x'].size
        self.DataDev['Np'].fill(Np)
        self.Args['Np'] = Np

    def fill_arr_randn(self, arr, mu=0, sigma=1):
        return self._generator_knl.fill_normal(ary=arr, queue=self.queue,
                                               mu=mu, sigma=sigma)

    def fill_arr_rand(self, arr, xmin=0, xmax=1):
        return self._generator_knl.fill_uniform(ary=arr, queue=self.queue,
                                                a=xmin,b=xmax)

    def _cumsum(self,arr_in):
        evnt, arr_tmp = cumsum(arr_in, return_event=True, queue=self.queue)
        arr_out = self.dev_arr(dtype=np.uint32, shape=arr_tmp.size+1)
        arr_out[0] = 0
        evnt.wait()
        arr_out[1:] = arr_tmp[:]
        return arr_out
