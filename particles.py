import numpy as np

from chimeraCL.methods.particles_methods_cl import ParticleMethodsCL
from chimeraCL.methods.particles_methods_cl import sqrt


class Particles(ParticleMethodsCL):
    def __init__(self, configs_in, comm=None):
        if comm is not None:
            self.import_comm(comm)
        self.set_global_working_group_size()

        self.DataDev = {}
        self.init_particle_methods()
        self._process_configs(configs_in)

        if self.Args['Nppc'] is not None:
            self.prepare_generator_data()

        self.send_args_to_dev()

    def sort_parts(self, grid):
        self.index_and_sum(grid)
        self.index_sort()

    def align_parts(self):
        self.align_and_damp(
            comps_align = ['x', 'y', 'z', 'px', 'py', 'pz', 'g_inv', 'w'],
            comps_simple_dump = ['indx_in_cell',
                                 'Ex', 'Ey', 'Ez',
                                 'Bx', 'By', 'Bz'])

    def _process_configs(self, configs_in):
        self.Args = configs_in

        self.Args['Np'] = 0

        if 'dt' not in self.Args:
            self.Args['dt'] = 1.

        if 'q' not in self.Args:
            self.Args['q'] = 1.

        if 'Nppc' not in self.Args:
            self.Args['Nppc'] = (2,2,4)

        self.Args['Nppc'] = np.array(self.Args['Nppc'],dtype=np.uint32)

        self.Args['dt_2'] = 0.5*self.Args['dt']
        self.Args['dt_inv'] = 1./self.Args['dt']
