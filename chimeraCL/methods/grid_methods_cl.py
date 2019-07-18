import numpy as np

from pyopencl import Program
from pyopencl import enqueue_marker, enqueue_barrier

from .generic_methods_cl import GenericMethodsCL
from .generic_methods_cl import compiler_options

from chimeraCL import __path__ as src_path
src_path = src_path[0] + '/kernels/'


class GridMethodsCL(GenericMethodsCL):
    def init_grid_methods(self):
        self.init_generic_methods()
        self.set_global_working_group_size()

        grid_sources = []
        grid_sources.append(''.join(open(src_path+"grid_generic.cl")\
              .readlines()) )
        grid_sources.append(''.join( \
              open(src_path+"grid_deposit_m"+str(self.Args['M'])+".cl")\
              .readlines() ) )

        grid_sources = self.block_def_str + ''.join(grid_sources)

        prg = Program(self.ctx, grid_sources).\
            build(options=compiler_options)

        self._divide_by_dv_d_knl = prg.divide_by_dv_d
        self._divide_by_dv_c_knl = prg.divide_by_dv_c
        self._treat_axis_d_knl = prg.treat_axis_d
        self._treat_axis_c_knl = prg.treat_axis_c
        self._warp_axis_m0_d_knl = prg.warp_axis_m0_d
        self._warp_axis_m1plus_c_knl = prg.warp_axis_m1plus_c

        self._depose_scalar_knl = prg.depose_scalar
        self._depose_vector_knl = prg.depose_vector
        self._gather_and_push_knl = prg.gather_and_push

        if 'vec_comps' not in self.Args:
            self.Args['vec_comps'] = self.Args['default_vec_comps']

    def depose_scalar(self, parts, src_scalar, dest_fld, charge):
        WGS, WGS_tot = self.get_wgs(self.Args['NxNr_4'])

        if parts.Args['Np'] <= 0:
            return

        part_str = ['sort_indx',] + self.Args['vec_comps'] + \
                    [src_scalar, 'cell_offset',]

        grid_str = ['Nx', 'Xmin', 'dx_inv',
                    'Nr', 'Rmin', 'dr_inv',
                    'NxNr_4']

        dest_fld += '_m'
        fld_str = [dest_fld + str(m) for m in range(self.Args['M']+1)]

        args_part = [parts.DataDev[arg].data for arg in part_str]
        args_grid = [self.DataDev[arg].data for arg in grid_str]
        args_fld = [self.DataDev[arg].data for arg in fld_str]

        args = args_part + [np.int8(charge),] + args_grid + args_fld

        for i_off in np.arange(4).astype(np.uint32):
            self._depose_scalar_knl(self.queue,
                                    (WGS_tot,),(WGS,),
                                    i_off, *args).wait()

    def depose_vector(self, parts, vec, factors, vec_fld, charge):

        part_str =  ['sort_indx',] + self.Args['vec_comps'] + vec + factors + \
                     ['cell_offset',]

        grid_str = ['Nx', 'Xmin', 'dx_inv',
                    'Nr', 'Rmin', 'dr_inv',
                    'NxNr_4']

        fld_str = []
        for m in range(self.Args['M']+1):
            for comp in self.Args['vec_comps']:
                fld_str.append(vec_fld + comp + '_m' + str(m))

        args_part = [parts.DataDev[arg].data for arg in part_str]
        args_grid = [self.DataDev[arg].data for arg in grid_str]
        args_fld = [self.DataDev[arg].data for arg in fld_str]

        args_dep = args_part + [np.int8(charge),] + args_grid + args_fld

        WGS, WGS_tot = self.get_wgs(self.Args['NxNr_4'])
        for i_off in np.arange(4).astype(np.uint32):
            self._depose_vector_knl(self.queue,
                                    (WGS_tot,),(WGS,),
                                    i_off, *args_dep).wait()


    def postproc_depose_scalar(self, fld):
        # Correct near axis deposition
        args_grid = [self.DataDev[fld+'_m'+str(m)].data \
                     for m in range(self.Args['M']+1)]

        WGS, WGS_tot = self.get_wgs(self.Args['Nx'])
        enqueue_barrier(self.queue)
        self._treat_axis_d_knl(self.queue, (WGS_tot,), (WGS,),
            args_grid[0], np.uint32(self.Args['Nx'])).wait()


        for m in range(1,self.Args['M']+1):
            self._treat_axis_c_knl(self.queue, (WGS_tot,), (WGS,),
                args_grid[m], np.uint32(self.Args['Nx'])).wait()

        # Divide by radius
        WGS, WGS_tot = self.get_wgs(self.Args['NxNr'])
        grid_str =  ['NxNr','Nx','dV_inv']
        grid_args = [self.DataDev[arg].data for arg in grid_str]

        enqueue_barrier(self.queue)
        self._divide_by_dv_d_knl(self.queue,(WGS_tot,), (WGS,),
                                 args_grid[0],*grid_args).wait()

        for m in range(1,self.Args['M']+1):
            self._divide_by_dv_c_knl(self.queue,(WGS_tot,), (WGS,),
                                     args_grid[m],*grid_args).wait()

    def postproc_depose_vector(self, vec_fld):
        args_raddiv_str =  ['NxNr','Nx','dV_inv']
        args_raddiv = [self.DataDev[arg].data for arg in args_raddiv_str]

        for fld in [vec_fld + comp for comp in self.Args['vec_comps']]:
            # Correct near axis deposition
            args_fld = [self.DataDev[fld+'_m'+str(m)].data \
                         for m in range(self.Args['M']+1)]

            WGS, WGS_tot = self.get_wgs(self.Args['Nx'])
            self._treat_axis_d_knl(self.queue, (WGS_tot,), (WGS,),
                args_fld[0], np.uint32(self.Args['Nx'])).wait()

            for m in range(1,self.Args['M']+1):
                self._treat_axis_c_knl(self.queue, (WGS_tot, ), (WGS, ),
                    args_fld[m], np.uint32(self.Args['Nx'])).wait()

            # Divide by radius
            WGS, WGS_tot = self.get_wgs(self.Args['NxNr'])
            enqueue_barrier(self.queue)
            self._divide_by_dv_d_knl(self.queue, (WGS_tot, ), (WGS, ),
                                      args_fld[0], *args_raddiv).wait()

            for m in range(1,self.Args['M']+1):
                self._divide_by_dv_c_knl(self.queue,(WGS_tot, ), (WGS, ),
                                         args_fld[m], *args_raddiv).wait()

    def preproc_project_vec(self, vec_fld):
        WGS, WGS_tot = self.get_wgs(self.Args['Nx'])
        for comp in self.Args['vec_comps']:
            fld = vec_fld + comp

            self._warp_axis_m0_d_knl(self.queue, (WGS_tot, ), (WGS, ),
                                     self.DataDev[fld+'_m0'].data,
                                     np.uint32(self.Args['Nx'])
                                    ).wait()

            for m in range(1,self.Args['M']+1):
                self._warp_axis_m1plus_c_knl(self.queue, (WGS_tot, ), (WGS, ),
                    self.DataDev[fld + '_m' + str(m)].data,
                    np.uint32(self.Args['Nx'])).wait()

    def _gather_and_push(self, parts, flds):
        part_str = ['x', 'y', 'z', 'px', 'py', 'pz', 'g_inv',
                    'sort_indx','cell_offset', 'FactorPush']

        grid_str = ['Nx', 'Xmin', 'dx_inv',
                    'Nr', 'Rmin', 'dr_inv',
                    'Nxm1Nrm1']

        fld_str = []
        for m in range(self.Args['M']+1):
            for fld in flds:
                for comp in self.Args['vec_comps']:
                    fld_str.append(fld + comp + '_m' + str(m))

        args_parts = [parts.DataDev[arg].data for arg in part_str]
        args_grid = [self.DataDev[arg].data for arg in grid_str]
        args_fld = [self.DataDev[arg].data for arg in fld_str]
        args_num_p = [np.uint32(parts.Args['Np']),
                      np.uint32(parts.Args['Np_stay'])]

        args = args_parts + args_num_p + args_grid + args_fld

        WGS, WGS_tot = self.get_wgs(parts.Args['Np'])
        self._gather_and_push_knl(self.queue, (WGS_tot, ), (WGS, ),
                                  *args).wait()
