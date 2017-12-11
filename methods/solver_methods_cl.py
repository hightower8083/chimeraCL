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

    def advance_fields(self, vecs):
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

            WGS, WGS_tot = self.get_wgs(self.Args['NxNrm1'])
            self._advance_e_g_m_knl(self.queue, (WGS_tot, ), (WGS, ),*args)
        enqueue_barrier(self.queue)
