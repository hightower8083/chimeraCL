import numpy as np

from chimeraCL.methods.grid_methods_cl import GridMethodsCL

class Grid(GridMethodsCL):
    def __init__(self, configs_in, comm):
        if comm is not None:
            self.import_comm(comm)

        self.set_global_working_group_size()

        self._process_configs(configs_in)

        if 'vec_comps' not in self.Args:
            self.Args['vec_comps'] = ['x','y','z']

        self.init_grid_methods()

        self.DataDev = {}
        self._init_grid_data_on_dev()
        self.send_args_to_dev()

    def depose_charge(self, species=[]):
        for m in range(self.Args['M']+1):
            self.set_to(self.DataDev['rho_m'+str(m)], 0)

        for parts in species:
            self.depose_scalar(parts, 'w', 'rho',
                               charge=parts.Args['charge'])

        self.postproc_depose_scalar('rho')

    def depose_currents(self, species=[]):

        current_args_str = []
        for m in range(self.Args['M']+1):
            for comp_str in self.Args['vec_comps']:
                current_args_str.append('J' + comp_str + '_m' + str(m))

        momentum_args_str = []
        for comp_str in self.Args['vec_comps']:
            momentum_args_str.append('p' + comp_str)

        for arg in current_args_str:
            self.set_to(self.DataDev[arg], 0)

        for parts in species:
            if 'Immobile' in parts.Args.keys():
                continue
            self.depose_vector(parts, momentum_args_str,
                               ['g_inv', 'w'], 'J',
                               charge=parts.Args['charge'])

        self.postproc_depose_vector('J')

    def project_fields(self, species=[]):

        flds_str = ['E', 'B']

        for fld in flds_str:
            self.preproc_project_vec(fld)

        for parts in species:

            if 'Immobile' in parts.Args.keys():
                continue

            flds_comps_str = []
            for fld_str in flds_str:
                for comp_str in self.Args['vec_comps']:
                    flds_comps_str.append(fld_str + comp_str)

            for arg in flds_comps_str:
                self.set_to(parts.DataDev[arg], 0)
            self.project_vec6(parts, flds_str, flds_str)

    def _process_configs(self, configs_in):
        self.Args = configs_in

        if 'M' not in self.Args:
            self.Args['M'] = 0

        self.Args['dx'] = (self.Args['Xmax']-self.Args['Xmin']) / \
                          (self.Args['Nx']-1)
        self.Args['dx_inv'] = 1./self.Args['dx']

        self.Args['dr'] = self.Args['Rmax'] / (self.Args['Nr']-1.5)
        self.Args['dr_inv'] = 1./self.Args['dr']

        if 'dt' not in self.Args:
            self.Args['dt'] = self.Args['dx']
        self.Args['dt_inv'] = 1.0/self.Args['dt']

        self.Args['Xgrid'] = self.Args['Xmin'] + self.Args['dx'] * \
            np.arange(self.Args['Nx'])

        self.Args['Rmin'] = -0.5*self.Args['dr']
        self.Args['Rgrid'] = self.Args['Rmin'] + self.Args['dr'] * \
            np.arange(self.Args['Nr'])
        self.Args['Rmax'] = self.Args['Rgrid'].max()

        self.Args['dV_inv'] = (self.Args['Rgrid'] > 0) \
                              / (2*np.pi*self.Args['dx']*self.Args['dr']\
                                 *self.Args['Rgrid'] )

        self.Args['NxNr'] = self.Args['Nr'] * self.Args['Nx']
        self.Args['Nxm1Nrm1'] = (self.Args['Nr']-1) * (self.Args['Nx']-1)
        self.Args['NxNrm1'] = (self.Args['Nr']-1) * self.Args['Nx']
        self.Args['NxNr_4'] = self.Args['Nr']//2 * self.Args['Nx']//2

        self.Args['dont_send'] = []
        self.Args['dont_keep'] = []


    def _init_grid_data_on_dev(self):

        flds_vec_str = ['E', 'B', 'J', 'G']
        flds_scl_str = ['rho', ]

        flds_comps_str = []
        for fld_str in flds_vec_str:
            for comp_str in self.Args['vec_comps']:
                flds_comps_str.append(fld_str + comp_str)

        flds_comps_str = flds_comps_str + flds_scl_str

        for arg in flds_comps_str:
            arg += '_m'
            self.DataDev[arg+'0'] = self.dev_arr(
                val=0, dtype=np.double,
                shape=(self.Args['Nr'], self.Args['Nx']))

            for m in range(1, self.Args['M']+1):
                self.DataDev[arg+str(m)] = self.dev_arr(
                    val=0, dtype=np.complex128,
                    shape=(self.Args['Nr'], self.Args['Nx']))
