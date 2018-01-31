import numpy as np


class Frame():
    def __init__(self, configs_in, comm=None):
        self._process_configs(configs_in)

    def _process_configs(self, configs_in):
        self.Args = configs_in

        if 'Steps' not in self.Args:
            self.Args['Steps'] = 1.

        if 'Velocity' not in self.Args:
            self.Args['Velocity'] = 0.

        if 'Ndump' not in self.Args:
            self.Args['Ndump'] = 0

        if 'dt' not in self.Args:
            self.Args['dt'] = 1
        if 'DensityProfiles' not in self.Args:
            self.Args['DensityProfiles'] = None

    def shift_grids(self, grids=[], steps=None):
        if steps is None:
            steps = self.Args['Steps']
        x_shift = steps * self.Args['dt'] * self.Args['Velocity']

        for grid in grids:
            for store in [grid.Args, grid.DataDev]:
                for arg in ['Xmax','Xmin','Xgrid']:
                    store[arg] += x_shift

    def inject_plasma(self, species, grid, steps=None):

        if steps is None:
            steps = self.Args['Steps']
        x_shift = steps * self.Args['dt'] * self.Args['Velocity']

        for specie in species:
            if specie.Args['Np'] == 0:
                specie.Args['right_lim'] = grid.Args['Xmax'] - x_shift

            inject_domain = {}
            inject_domain['Xmin'] = specie.Args['right_lim']
            inject_domain['Xmax'] = inject_domain['Xmin'] + x_shift
            inject_domain['Rmin'] = grid.Args['Rmin']*(grid.Args['Rmin']>0)
            inject_domain['Rmax'] = grid.Args['Rmax']

            specie.make_new_domain(inject_domain,
                density_profiles=self.Args['DensityProfiles'])

            if 'InjectorSource' in specie.Args.keys():
                specie.add_new_particles(specie.Args['InjectorSource'])
            else:
                specie.add_new_particles()

        for specie in species:
            specie.free_added()
            specie.sort_parts(grid=grid, Ndump=self.Args['Ndump'])
            specie.align_parts()

            Num_ppc = np.int32(np.prod(specie.Args['Nppc'])+1)
            x_max = specie.DataDev['x'][-Num_ppc:].get().max()

            specie.Args['right_lim'] = x_max + 0.5*specie.Args['ddx']
