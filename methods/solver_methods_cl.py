import numpy as np

from pyopencl import Program
from pyopencl import enqueue_marker, enqueue_barrier
from pyopencl.array import empty_like

from .generic_methods_cl import GenericMethodsCL
from .generic_methods_cl import compiler_options

from chimeraCL import __path__ as src_path
src_path = src_path[0] + '/kernels/'


class SolverMethodsCL(GenericMethodsCL):
    def init_solver_methods(self):
        solver_sources = []
        solver_sources.append(''.join(open(src_path+"solver_ms_pic.cl")\
              .readlines()) )

        solver_sources = self.block_def_str + ''.join(solver_sources)

        prg = Program(self.ctx, solver_sources).\
            build(options=compiler_options)

        self._advance_e_g_m_knl = prg.advance_e_g_m
        self._profile_edges_c_knl = prg.profile_edges_c
        self._profile_edges_d_knl = prg.profile_edges_d

        if 'DampCells' in self.Args:
            self._init_field_damping()

    def _init_field_damping(self):
        N = int(self.Args['DampCells'])
        z = np.arange(2*N)
        z_shft = 3.*(z-N+1)/(N+1)
        DmpProf = (z < 4.*N/3) * (z >= N) * np.sin(0.5 * np.pi * z_shft)**2 \
                  + (z >= 4.*N/3)

        self.Args['DampProfile'] = DmpProf
        self.Args['dont_keep'].append('DampProfile')
        self.Args['dont_send'].append('DampCells')

    def advance_fields(self, vecs):
        WGS, WGS_tot = self.get_wgs(self.Args['NxNrm1'])
        for m in range(self.Args['M']+1):
            mstr = '_m'+str(m)
            solver_str = ['NxNrm1', 'dt_inv',
                          'MxSlv_cos(wdt)' + mstr,
                          'MxSlv_sin(wdt)*w' + mstr,
                          'MxSlv_1/w**2' + mstr]

            mstr = '_fb_m'+str(m)
            fld_str = []
            for v in vecs:
                for c in self.Args['vec_comps']:
                    fld_str.append(v+c+mstr)

            args_solver = [self.DataDev[arg].data for arg in solver_str]
            args_fld = [self.DataDev[arg].data for arg in fld_str]

            args = args_solver + args_fld
            self._advance_e_g_m_knl(self.queue, (WGS_tot, ), (WGS, ),*args).wait()

    def profile_edges(self, flds):
        WGS, WGS_tot = self.get_wgs(self.Args['NxNr'])

        if 'DampCells' not in self.Args:
            return

        for fld in flds:
            for comp in self.Args['vec_comps']:
                for m in range(self.Args['M']+1):
                    fld_str = fld + comp + '_m' + str(m)

                    if self.DataDev[fld_str].dtype==np.double:
                        profiler = self._profile_edges_d_knl
                    elif self.DataDev[fld_str].dtype==np.complex128:
                        profiler = self._profile_edges_c_knl

                    profiler(self.queue, (WGS_tot, ), (WGS, ),
                             self.DataDev[fld_str].data,
                             self.DataDev['DampProfile'].data,
                             np.uint32(self.Args['NxNr']),
                             np.uint32(self.Args['Nx']),
                             np.uint32(2*self.Args['DampCells'])
                            ).wait()
