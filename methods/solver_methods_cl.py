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

    def add_gausian_pulse(self, laser):

        k0 = 2*np.pi*laser['k0']
        a0 = laser['a0']

        Lx = laser['Lx']
        R = laser['R']
        x0 = laser['x0']
        x_foc = laser['x_foc']
        X_focus = x0 - x_foc

        Xgrid,Rgrid = self.Args['Xgrid'],self.Args['Rgrid'] # sin phase
        kx,w = self.Args['kx'][None,:], self.Args['w_m0']

        self.DataDev['Ez_m0'][:] = a0 * np.sin(k0*(Xgrid[None,:]-x0)) \
              * np.exp(-(Xgrid[None,:]-x0)**2/Lx**2 - Rgrid[:,None]**2/R**2) \
              * (abs(Rgrid[:,None]) < 3.5*R) * (abs(Xgrid[None,:]-x0) < 3.5*Lx)

        self.fb_transform(scals=['Ez',],dir=0)
        EE = self.DataDev['Ez_fb_m0'].get()

        DT = -1.j*w*np.sign(kx + (kx==0))
        GG  = DT*EE

        EE_tmp = np.cos(w*X_focus)*EE + np.sin(w*X_focus)/w*GG
        GG = -w*np.sin(w*X_focus)*EE + np.cos(w*X_focus)*GG
        EE = EE_tmp
        EE *= np.exp(1.j * kx * X_focus)
        GG *= np.exp(1.j * kx * X_focus)

        self.DataDev['Ez_fb_m0'][:] = EE
        self.DataDev['Gz_fb_m0'][:] = GG

