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

        if 'dt' not in self.Args:
            self.Args['dt'] = 1

    def shift_grids(self, grids=[], steps=None):
        if steps is None:
            steps = self.Args['Steps']
        x_shift = steps * self.Args['dt'] * self.Args['Velocity']

        for grid in grids:
            for store in [grid.Args, grid.DataDev]:
                for arg in ['Xmax','Xmin','Xgrid']:
                    store[arg] += x_shift

    def inject_plasma(self, species=[], grid=None, steps=None):
        if steps is None:
            steps = self.Args['Steps']
        x_shift = steps * self.Args['dt'] * self.Args['Velocity']

        for specie in species:
            if specie.Args['Np'] == 0:
                specie.Args['right_lim'] = grid.Args['Xmax'] - x_shift

            inject_domain = {}
            inject_domain['Xmin'] = specie.Args['right_lim']
            inject_domain['Xmax'] = inject_domain['Xmin'] + x_shift
            inject_domain['Rmin'] = grid.Args['Rmin']
            inject_domain['Rmax'] = grid.Args['Rmax']

            specie.add_particles(domain_in=inject_domain)
            specie.sort_parts(grid=grid)
            specie.align_parts()
            specie.Args['right_lim'] = specie.DataDev['x'][-1].get() \
                                       + 0.5*specie.Args['ddx']
