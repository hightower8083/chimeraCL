import numpy as np


class PIC_loop:
    def __init__(self, solvers, species, frames):
        self.solvers = solvers
        self.mainsolver = self.solvers[0]
        self.species = species
        self.frames = frames
        self.it = 0

    def step(self):
        for frame in self.frames:
            if np.mod(self.it, frame.Args['Steps']) == 0:
                frame.shift_grids(grids=self.solvers)
                frame.inject_plasma(species=self.species, grid=self.mainsolver)

        if np.mod(self.it, frame.Args['Steps']) == 0:
            for parts in self.species:
                parts.free_mp()

        for parts in self.species:
            parts.push_coords(mode='half')
            parts.sort_parts(grid=self.mainsolver)

        for solver in self.solvers:
            solver.depose_currents(species=self.species)

        for parts in self.species:
            parts.push_coords(mode='half')
            parts.sort_parts(grid=self.mainsolver)

        for solver in self.solvers:
            solver.depose_charge(species=self.species)
            solver.fb_transform(scals=['rho', ], vects=['J', ], dir=0)
            solver.fields_smooth(flds=['rho','Jx','Jy','Jz'])

            for m in range(0,solver.Args['M']+1):
                for comp in solver.Args['vec_comps']:
                    arg_str = comp +'_fb_m'+str(m)
                    solver.DataDev['dN0'+arg_str][:] = \
                        solver.DataDev['dN1'+arg_str]

            solver.field_grad('rho','dN1')

            solver.push_fields()
            solver.damp_fields()
            solver.restore_B_fb()

            solver.fb_transform(vects=['E', 'B'], dir=1)
            solver.gather_and_push(species=self.species)

        for parts in self.species:
            parts.sort_parts(grid=self.mainsolver)

        self.it +=1
        return self.it
