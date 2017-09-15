import numpy as np

from chimeraCL.methods.particles_methods_cl import ParticleMethodsCL
from chimeraCL.methods.particles_methods_cl import sqrt


class Particles(ParticleMethodsCL):
    def __init__(self, configs_in, comm=None):
        if comm is not None:
            self.comm = comm
            self.queue = comm.queue
            self.ctx = comm.ctx
            self.thr = comm.thr
            self.dev_type = comm.dev_type
            self.plat_name = comm.plat_name

        self.init_particle_methods()
        self._process_configs(configs_in)
        self.send_args_to_dev()

    def make_parts(self, beam_in):
        Np = beam_in['Np']
        for arg in ['x', 'y', 'z', 'px', 'py', 'pz', 'g_inv', 'w',
                    'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']:
            self.DataDev[arg] = self.dev_arr(val=0, shape=Np, dtype=np.double)

        for arg in ['x', 'y', 'z']:
            self.fill_arr_randn(self.DataDev[arg],
                                mu=beam_in[arg+'_c'],
                                sigma=beam_in['L'+arg])

        for arg in ['px', 'py', 'pz']:
            self.fill_arr_randn(self.DataDev[arg],
                                mu=beam_in[arg+'_c'],
                                sigma=beam_in['d'+arg])

        self.DataDev['g_inv'] = 1./sqrt(1. +
                                        self.DataDev['px']*self.DataDev['px'] +
                                        self.DataDev['py']*self.DataDev['py'] +
                                        self.DataDev['pz']*self.DataDev['pz'])

        self.DataDev['w'][:] = self.Args['q']
        self.DataDev['indx_in_cell'] = self.dev_arr(val=0, dtype=np.uint32,
                                                    shape=Np)
        self.reset_num_parts()

    def sort_parts(self, grid):
        self.index_and_sum(grid)
        self.index_sort()

    def align_parts(self):
        self.align_and_damp(['x', 'y', 'z', 'px', 'py', 'pz', 'g_inv', 'w'],
                            ['indx_in_cell', 'Ex', 'Ey', 'Ez',
                             'Bx', 'By', 'Bz'])

    def _process_configs(self, configs_in):
        self.Args = configs_in

        self.Args['Np'] = 0

        if 'dt' not in self.Args:
            self.Args['dt'] = 1.

        if 'q' not in self.Args:
            self.Args['q'] = 1.

        self.Args['dt_2'] = 0.5*self.Args['dt']
        self.Args['dt_inv'] = 1.0/self.Args['dt']

    def _send_grid_to_dev(self):
        self.DataDev = {}

        for arg in ['Np', ]:
            self.DataDev[arg] = self.dev_arr(val=self.Args[arg],
                                             dtype=np.uint32)

        for arg in ['dt_2', 'dt_inv']:
            self.DataDev[arg] = self.dev_arr(val=self.Args[arg],
                                             dtype=np.double)
