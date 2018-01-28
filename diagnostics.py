"""
    Minimalistic diagnostics module for ChimeraCL.
    Outputs fields and species to the hdf5 format using simple
    path convention:
      - fields are in "/data/fields/" (e.g. "/data/fields/rho")
      - species are in "/data/species/species_ID" (ID is an integer
        position of a given species in the list of species=[])
      - complementary simulation information is in "/data/info/"

    NB: Electic and magnetic fields are in dimensionaless units;
        Charge density is normalised to critical density for the
          space normalisation length;
        species weights are rescaled to pC/lambda[um], where lambda[um] is
          a space normalisation in microns;
"""

import numpy as np
import h5py
import os

class Diagnostics:
    def __init__(self, configs_in, solver, species=[], frame=None,
                 path='diags', dtype_flds=np.float32, dtype_parts=np.float64):

        self.Args = configs_in
        self.solver = solver
        self.species = species
        self.frame = frame
        self.dtype_flds = dtype_flds
        self.dtype_parts = dtype_parts

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
            self.Args['Species'] = {'Components':[],}

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
            if 'Selections' in self.Args['Species']:
                select_mask = np.ones(part.Args['Np'], dtype=np.uint8)
                for select in  self.Args['Species']['Selections']:
                    comp_select, vmin, vmax = select
                    comp_val = part.DataDev[comp_select].get()
                    if vmin is not None:
                        select_mask *= (comp_val>vmin).astype(np.uint8)
                    if vmax is not None:
                        select_mask *= (comp_val<vmax).astype(np.uint8)
                indx = np.nonzero(select_mask)
            else:
                indx = None

            for part_comp in self.Args['Species']['Components']:
                if part.Args['Np']==0:
                    comp_vals = np.array([], dtype=self.dtype_parts)
                elif indx is None:
                    comp_vals = part.DataDev[part_comp].\
                        get().astype(self.dtype_parts)
                else:
                    comp_vals = part.DataDev[part_comp].\
                        map_to_host()[indx].astype(self.dtype_parts)

                if part_comp =='w':
                    comp_vals *= self.dtype_parts(self.Args['w2pC'])

                self.record[h5_path+part_comp] = comp_vals

    def add_generic_info(self):
        h5_path = self.base_str + self.info_str
        for key in self.generic_keys:
            self.record[h5_path + key] = self.solver.Args[key]
        if self.frame is None:
            self.record[h5_path + 'FrameVelocity'] = 0.
        else:
            self.record[h5_path + 'FrameVelocity'] = self.frame.Args['Velocity']

    def add_field(self, fld):
        h5_path = self.base_str + self.flds_str + fld

        if fld=='rho' or fld[0]=='J':
            self.solver.depose_charge(self.species)
            self.solver.fb_transform(scals=[fld, ], dir=0)
            self.solver.fields_smooth(flds=[fld, ])

        self.solver.fb_transform(scals=[fld, ], dir=1)

        fld_m = self.solver.DataDev[fld + '_m0'].get()[None,:] # mod
        fld_stack = [fld_m.astype(self.dtype_flds), ]

        for m in range(1, self.solver.Args['M']+1, 1):
            fld_m = self.solver.DataDev[fld + '_m' + str(m)]\
                    .get()[None,:] # mod
            fld_stack.append(fld_m.real.astype(self.dtype_flds))
            fld_stack.append(fld_m.imag.astype(self.dtype_flds))

        self.record[h5_path] = np.concatenate(fld_stack, axis=0)
