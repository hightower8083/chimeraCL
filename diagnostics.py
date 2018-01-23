import numpy as np
import h5py
import os

class Diagnostics:
    def __init__(self, configs_in, solver, species=[], path='diags'):
        self.Args = configs_in
        self.solver = solver
        self.species = species

        self.base_str = '/data/'
        self.flds_str = 'fields/'
        self.parts_str = 'species/'
        self.info_str = 'info/'
        self.generic_keys = ['Xgrid', 'Rgrid',
                             'dx', 'dr', 'dt',
                             'Nx', 'Nr','M']

        self.path = os.getcwd() + '/' + path + '/'

        if os.path.exists(self.path) == False:
            os.makedirs(self.path)
        else:
            for fl in os.listdir(self.path):
                os.remove(self.path+fl)

        if 'ScalarFields' not in self.Args:
            self.Args['ScalarFields'] = []

        if 'VectorFields' not in self.Args:
            self.Args['VectorFields'] = []

        if 'Species' not in self.Args:
            self.Args['Species'] = []


    def make_record(self, it):
        if np.mod(it, self.Args['Interval']) != 0:
            return

        it_str = str(it)
        while len(it_str)<9: it_str = '0' + it_str

        self.record = h5py.File(self.path + it_str + '.h5', 'w')

        self.record[self.base_str + self.info_str + 'iteration'] = it

        self.add_generic_info()

        for fld in self.Args['ScalarFields']:
            self.add_field(fld)

        for fld in self.Args['VectorFields']:
            for comp in ['x', 'y', 'z']:
                self.add_field(fld+comp)

        self.add_species()
        self.record.close()

    def add_species(self):
        for species_index in np.arange(len(self.species)):
            part = self.species[species_index]
            specie_name = 'species_' + str(species_index) + '/'
            h5_path = self.base_str + self.parts_str + specie_name
            for part_comp in self.Args['Species']:
                self.record[h5_path+part_comp] = part.DataDev[part_comp].get()

    def add_generic_info(self):
        h5_path = self.base_str + self.info_str
        for key in self.generic_keys:
            self.record[h5_path + key] = self.solver.Args[key]

    def add_field(self, fld):
        h5_path = self.base_str + self.flds_str + fld

        if fld=='rho' or fld[0]=='J':
            self.solver.depose_charge(self.species)
            self.solver.fb_transform(scals=[fld, ], dir=0)
            self.solver.fields_smooth(flds=[fld, ])

        self.solver.fb_transform(scals=[fld, ], dir=1)

        fld_m = self.solver.DataDev[fld + '_m0'].get()[None,1:]
        fld_stack = [fld_m, ]

        for m in range(1, self.solver.Args['M']+1, 1):
            fld_m = self.solver.DataDev[fld + '_m' + str(m)].get()[None,1:]
            fld_stack.append(fld_m.real)
            fld_stack.append(fld_m.imag)

        self.record[h5_path] = np.concatenate(fld_stack, axis=0)
