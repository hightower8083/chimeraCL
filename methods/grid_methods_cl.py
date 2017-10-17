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

        self._divide_by_dv_dbl_knl = prg.divide_by_dv_dbl
        self._divide_by_dv_clx_knl = prg.divide_by_dv_clx
        self._treat_axis_dbl_knl = prg.treat_axis_dbl
        self._treat_axis_clx_knl = prg.treat_axis_clx

        self._depose_scalar_knl = prg.depose_scalar
        self._depose_vector_knl = prg.depose_vector
        self._project_scalar_knl = prg.project_scalar
        self._project_vec6_knl = prg.project_vec6

    def depose_scalar(self, parts, sclr, fld):
        WGS, WGS_tot = self.get_wgs(self.Args['NxNr_4'])

        if parts.Args['Np'] <= 0:
            return

        part_str = ['sort_indx','x','y','z',
                     sclr,'cell_offset',]
        grid_str = ['Nx', 'Xmin', 'dx_inv',
                     'Nr', 'Rmin', 'dr_inv',
                     'NxNr_4']
        fld_str = [fld+'_m'+str(m) for m in range(self.Args['M']+1)]

        args_part = [parts.DataDev[arg].data for arg in part_str]
        args_grid = [self.DataDev[arg].data for arg in grid_str]
        args_fld = [self.DataDev[arg].data for arg in fld_str]

        args = args_part + args_grid + args_fld

        evnt = enqueue_marker(self.queue)
        for i_off in np.arange(4).astype(np.uint32):
            evnt = self._depose_scalar_knl(self.queue,
                                           (WGS_tot,),(WGS,),
                                           i_off, *args,
                                           wait_for = [evnt,])

    def postproc_depose_scalar(self, fld):
        # Correct near axis deposition
        args_grid = [self.DataDev[fld+'_m'+str(m)].data \
                     for m in range(self.Args['M']+1)]

        WGS, WGS_tot = self.get_wgs(self.Args['Nx'])
        enqueue_barrier(self.queue)
        self._treat_axis_dbl_knl(self.queue,
                                 (WGS_tot,), (WGS,),
                                 args_grid[0], np.uint32(self.Args['Nx']))

        for m in range(1,self.Args['M']+1):
            self._treat_axis_clx_knl(self.queue,
                                     (WGS_tot,), (WGS,),
                                     args_grid[m], np.uint32(self.Args['Nx']))
        # Divide by radius
        WGS, WGS_tot = self.get_wgs(self.Args['NxNr'])
        grid_str =  ['NxNr','Nx','dV_inv']
        grid_args = [self.DataDev[arg].data for arg in grid_str]

        enqueue_barrier(self.queue)
        self._divide_by_dv_dbl_knl(self.queue,(WGS_tot,), (WGS,),
                             args_grid[0],*grid_args)

        for m in range(1,self.Args['M']+1):
            self._divide_by_dv_clx_knl(self.queue,(WGS_tot,), (WGS,),
                                 args_grid[m],*grid_args)
        enqueue_barrier(self.queue)

    def project_scalar(self, parts, sclr, fld):
        part_str = ['sort_indx',sclr,'x','y','z','cell_offset']

        grid_str = ['Nx', 'Xmin', 'dx_inv',
                     'Nr', 'Rmin', 'dr_inv',
                     'Nxm1Nrm1']

        fld_str = [fld+'_m'+str(m) for m in range(self.Args['M']+1)]

        args_parts = [parts.DataDev[arg].data for arg in part_str]
        args_grid = [self.DataDev[arg].data for arg in grid_str]
        args_fld = [self.DataDev[arg].data for arg in fld_str]

        args = args_parts + args_grid + args_fld

        WGS, WGS_tot = self.get_wgs(self.Args['Nxm1Nrm1'])
        evnt = self._project_scalar_knl(self.queue, (WGS_tot,),(WGS,),*args)

    def depose_vector(self, parts, vec, factors,vec_fld):
        part_str =  ['sort_indx','x','y','z'] + vec + factors + \
                     ['cell_offset']

        grid_str = ['Nx', 'Xmin', 'dx_inv',
                     'Nr', 'Rmin', 'dr_inv',
                     'NxNr_4']

        fld_str = []
        for m in range(self.Args['M']+1):
            for comp in ('x','y','z'):
                fld_str.append(vec_fld + comp + '_m' + str(m))

        args_part = [parts.DataDev[arg].data for arg in part_str]
        args_grid = [self.DataDev[arg].data for arg in grid_str]
        args_fld = [self.DataDev[arg].data for arg in fld_str]

        args_dep = args_part + args_grid + args_fld

        WGS, WGS_tot = self.get_wgs(self.Args['NxNr_4'])
        evnt = enqueue_marker(self.queue)
        for i_off in np.arange(4).astype(np.uint32):
            evnt = self._depose_vector_knl(self.queue,
                                           (WGS_tot,),(WGS,),
                                           i_off, *args_dep,
                                           wait_for = [evnt,])
        enqueue_barrier(self.queue)

    def postproc_depose_vector(self,vec_fld):
        args_raddiv_str =  ['NxNr','Nx','Rgrid_inv']
        args_raddiv = [self.DataDev[arg].data for arg in args_raddiv_str]

        for fld in [vec_fld + comp for comp in ('x','y','z')]:
            # Correct near axis deposition
            args_fld = [self.DataDev[fld+'_m'+str(m)].data \
                         for m in range(self.Args['M']+1)]

            WGS, WGS_tot = self.get_wgs(self.Args['Nx'])
            self._treat_axis_dbl_knl(self.queue, (WGS_tot,), (WGS,),
                                     args_fld[0], np.uint32(self.Args['Nx']))

            for m in range(1,self.Args['M']+1):
                self._treat_axis_clx_knl(self.queue, (WGS_tot, ), (WGS, ),
                                         args_fld[m],
                                         np.uint32(self.Args['Nx']))

            # Divide by radius
            WGS, WGS_tot = self.get_wgs(self.Args['NxNr'])
            enqueue_barrier(self.queue)
            self._divide_by_dv_dbl_knl(self.queue, (WGS_tot, ), (WGS, ),
                                      args_fld[0], *args_raddiv)

            for m in range(1,self.Args['M']+1):
                self._divide_by_dv_clx_knl(self.queue,(WGS_tot, ), (WGS, ),
                                          args_fld[m], *args_raddiv)
            enqueue_barrier(self.queue)

    def project_vec6(self, parts, vecs, flds):
        part_str = ['sort_indx',] + \
                   [vecs[0] + comp for comp in ('x','y','z')] + \
                   [vecs[1] + comp for comp in ('x','y','z')] + \
                   ['x','y','z','cell_offset',]

        grid_str = ['Nx', 'Xmin', 'dx_inv',
                    'Nr', 'Rmin', 'dr_inv',
                    'Nxm1Nrm1']

        fld_str = []
        for m in range(self.Args['M']+1):
            for fld in flds:
                for comp in ('x','y','z'):
                    fld_str.append(fld + comp + '_m' + str(m))

        args_parts = [parts.DataDev[arg].data for arg in part_str]
        args_grid = [self.DataDev[arg].data for arg in grid_str]
        args_fld = [self.DataDev[arg].data for arg in fld_str]

        args = args_parts + args_grid + args_fld

        WGS, WGS_tot = self.get_wgs(self.Args['Nxm1Nrm1'])
        evnt = self._project_vec6_knl(self.queue, (WGS_tot, ), (WGS, ),*args)
