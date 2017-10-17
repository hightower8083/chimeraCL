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

    def shift(self, species=[], grids=[], steps=None):
        if steps is None:
            steps = self.Args['Steps']
        x_shift = steps * self.Args['dt'] * self.Args['Velocity']

        for grid in grids:
            for store in [grid.Args, grid.DataDev]:
                for arg in ['Xmax','Xmin','Xgrid']:
                    store[arg] += x_shift

        for specie in species:
            if specie.Args['Np']==0:
                return
            specie.sort_parts(grid=grids[0])
            specie.align_parts()
