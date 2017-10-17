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

        self.send_args_to_dev()

    def sort_parts(self, grid):
        if self.Args['Np'] == 0:
            return

        self.index_and_sum(grid)
        self.index_sort()

    def add_particles(self, domain_in=None, beam_in=None):
        if domain_in is not None:
            self.make_new_domain(domain_in)
        elif beam_in is not None:
            self.make_new_beam(beam_in)

        self.add_new_particles()

    def align_parts(self):
        if self.Args['Np'] == 0:
            return

        self.align_and_damp(
            comps_align = ['x', 'y', 'z', 'px', 'py', 'pz', 'g_inv', 'w'],
            comps_simple_dump = ['indx_in_cell',])

    def _process_configs(self, configs_in):
        self.Args = configs_in

        self.Args['Np'] = 0

        if 'dt' not in self.Args:
            self.Args['dt'] = 1.

        if 'charge' not in self.Args:
            self.Args['charge'] = -1.

        if 'mass' not in self.Args:
            self.Args['mass'] = -1.

        if 'dens' not in self.Args:
            self.Args['dens'] = 1.

        if 'Nppc' in self.Args:
            self.Args['Nppc'] = np.array(self.Args['Nppc'],dtype=np.uint32)

            self.Args['w0'] = 2 * np.pi * self.Args['dx'] * self.Args['dr'] \
                              * self.Args['charge'] * self.Args['dens'] \
                              / np.prod(self.Args['Nppc'])
            self.Args['ddx'] = self.Args['dx']/self.Args['Nppc'][0]
        else:
            self.Args['w0'] = self.Args['charge']/self.Args['Np']
            self.Args['ddx'] = 1

        self.Args['right_lim'] = 0.0

        self.Args['dt_2'] = 0.5*self.Args['dt']
        self.Args['dt_inv'] = 1./self.Args['dt']

        args_strs =  ['x', 'y', 'z', 'px', 'py', 'pz', 'w','g_inv']

        for arg in args_strs:
            self.DataDev[arg] = self.dev_arr(shape=0, dtype=np.double)
