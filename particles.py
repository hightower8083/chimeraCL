import numpy as np

from chimeraCL.methods.particles_methods_cl import ParticleMethodsCL
from chimeraCL.methods.particles_methods_cl import sqrt
from pyopencl.tools import MemoryPool, ImmediateAllocator

class Particles(ParticleMethodsCL):
    def __init__(self, configs_in, comm=None):
        if comm is not None:
            self.import_comm(comm)

        self.set_global_working_group_size()

        self.DataDev = {}
        self.init_particle_methods()
        self._process_configs(configs_in)

        self.send_args_to_dev()
        self._init_data_on_dev()

    def sort_parts(self, grid):
        if self.Args['Np'] == 0:
            self.flag_sorted = True

        if self.flag_sorted == False:
            self.index_sort(grid)
            self.flag_sorted = True

    def add_particles(self, domain_in=None, beam_in=None, source=None):
        # To be removed
        if source is None:
            if domain_in is not None:
                self.make_new_domain(domain_in)
            elif beam_in is not None:
                self.make_new_beam(beam_in)
            self.add_new_particles()
        else:
            self.add_new_particles(source=source)

    def align_parts(self):
        if self.Args['Np'] == 0:
            return
        if 'Immobile' in self.Args.keys():
            comps_align = ['x', 'y', 'z','w']
        else:
            comps_align = ['x', 'y', 'z', 'px', 'py', 'pz',
                           'g_inv', 'w']

        self.align_and_damp(comps_align=comps_align)

    def _process_configs(self, configs_in):
        self.Args = configs_in

        self.Args['Np'] = 0
        self.Args['Np_stay'] = 0

        if 'dt' not in self.Args:
            self.Args['dt'] = 1.

        if 'dx' not in self.Args:
            self.Args['dx'] = 1.

        if 'dr' not in self.Args:
            self.Args['dr'] = 1.

        if 'charge' not in self.Args:
            self.Args['charge'] = -1.

        if 'mass' not in self.Args:
            self.Args['mass'] = 1.

        if 'dens' not in self.Args:
            self.Args['dens'] = 1.

        dt, q_s, m_s, n_s, dx, dr = [self.Args[arg] for arg in ['dt',
            'charge', 'mass', 'dens', 'dx', 'dr']]

        self.Args['dt_2'] = 0.5 * dt

        if 'Nppc' in self.Args:
            self.Args['Nppc'] = np.array(self.Args['Nppc'],dtype=np.uint32)
            Num_ppc = np.prod(self.Args['Nppc'])
            self.Args['w0'] = 2 * np.pi * dx * dr * n_s / Num_ppc
            self.Args['ddx'] = dx / self.Args['Nppc'][0]
        else:
            self.Args['ddx'] = 1.

        self.Args['FactorPush'] = 2 * np.pi * dt * q_s / m_s

        self.Args['right_lim'] = 0.0

        self.flag_sorted = False

        self.Args['dont_send'] = ['InjectorSource','charge','mass',
                                  'dens','Immobile']
        self.Args['dont_keep'] = []

    def _init_data_on_dev(self):
        if 'Immobile' in self.Args.keys():
            args_strs =  ['x', 'y', 'z', 'w']
        else:
            args_strs =  ['x', 'y', 'z', 'px', 'py', 'pz', 'w','g_inv']

        for arg in args_strs:
            self.DataDev[arg] = self.dev_arr(shape=0,dtype=np.double)

        for arg in ['cell_offset', 'indx_in_cell',
                    'sort_indx', 'sum_in_cell']:
            allocator = ImmediateAllocator(self.comm.queue)
            self.DataDev[arg + '_mp'] = MemoryPool(allocator)

