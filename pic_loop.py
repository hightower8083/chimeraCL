import numpy as np
from time import time
from pyopencl import enqueue_barrier

loop_steps = ['frame', 'push-x', 'sort', 'depose',
              'transform', 'smooth', 'data_copy',
              'grad', 'push-eb', 'damp-eb', 'restore_B',
              'gather', 'push-p']

def timer_plot(Timer):
    import matplotlib.pyplot as plt
    import random

    fig, ax = plt.subplots(figsize=(16,5))
    Timer_keys = list(Timer.keys())
    Timer_values = [Timer[key] for key in Timer_keys]
    ind = np.arange(1,len(Timer_keys)+1)
    bars = plt.bar(ind, Timer_values)

    ax.set_xticks(ind)
    ax.set_xticklabels(Timer_keys)

    cnames = 2*['b', 'g', 'r', 'c', 'm', 'y', 'k',]

    random.shuffle(cnames)
    color_ind = 0
    for bar in bars:
        bar.set_facecolor(cnames[color_ind])
        color_ind +=1



class PIC_loop:
    def __init__(self, solvers, species, frames, timit=False):
        self.solvers = solvers
        self.mainsolver = self.solvers[0]
        self.species = species
        self.frames = frames
        self.timit = timit
        self.it = 0

        if self.timit is True:
            self.t_start = 0
            self.Timer = {}
            for str in loop_steps:
                self.Timer[str] = 0

    def timer_start(self):
        if self.timit is True:
            self.t_start = time()

    def timer_record(self, method_str):
        if self.timit is True:
            enqueue_barrier(self.mainsolver.comm.queue)
            self.Timer[method_str] += time() - self.t_start

    def step(self):

        self.timer_start()
        for frame in self.frames:
            if np.mod(self.it, frame.Args['Steps']) == 0:
                frame.shift_grids(grids=self.solvers)
                frame.inject_plasma(species=self.species, grid=self.mainsolver)

                for parts in self.species:
                    parts.free_mp()
        self.timer_record('frame')

        for parts in self.species:
            self.timer_start()
            parts.push_coords(mode='half')
            self.timer_record('push-x')

            self.timer_start()
            parts.sort_parts(grid=self.mainsolver)
            self.timer_record('sort')

        self.timer_start()
        for solver in self.solvers:
            solver.depose_currents(species=self.species)
        self.timer_record('depose')

        for parts in self.species:
            self.timer_start()
            parts.push_coords(mode='half')
            self.timer_record('push-x')

            self.timer_start()
            parts.sort_parts(grid=self.mainsolver)
            self.timer_record('sort')

        for solver in self.solvers:
            self.timer_start()
            solver.depose_charge(species=self.species)
            self.timer_record('depose')

            self.timer_start()
            solver.fb_transform(scals=['rho', ], vects=['J', ], dir=0)
            self.timer_record('transform')

            self.timer_start()
            solver.fields_smooth(flds=['rho','Jx','Jy','Jz'])
            self.timer_record('smooth')

            self.timer_start()
            for m in range(0,solver.Args['M']+1):
                for comp in solver.Args['vec_comps']:
                    arg_str = comp +'_fb_m'+str(m)
                    solver.DataDev['dN0'+arg_str][:] = \
                        solver.DataDev['dN1'+arg_str]
            self.timer_record('data_copy')

            self.timer_start()
            solver.field_grad('rho','dN1')
            self.timer_record('grad')

            self.timer_start()
            solver.push_fields()
            self.timer_record('push-eb')

            self.timer_start()
            solver.damp_fields()
            self.timer_record('damp-eb')

            self.timer_start()
            solver.restore_B_fb()
            self.timer_record('restore_B')

            self.timer_start()
            solver.fb_transform(vects=['E', 'B'], dir=1)
            self.timer_record('transform')

            self.timer_start()
            solver.project_fields(species=self.species)
            self.timer_record('gather')

        for parts in self.species:
            self.timer_start()
            parts.push_veloc()
            self.timer_record('push-p')

            #parts.sort_parts(grid=self.mainsolver)

        self.it +=1
        return self.it
