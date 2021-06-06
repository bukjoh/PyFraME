# Copyright (C) 2017-2021  Jógvan Magnus Haugaard Olsen and Peter Reinholdt
#
# This file is part of PyFraME.
#
# PyFraME is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyFraME is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyFraME.  If not, see <https://www.gnu.org/licenses/>.
#

from .fragments import FragmentDict
from .writers import InputWriters


__all__ = ['CoreRegion', 'RegionDict', 'Region']


class CoreRegion(object):

    """Container for region attributes and methods"""

    def __init__(self, fragments, **kwargs):
        self._fragments = fragments
        self._program = 'dalton'
        self._method = 'DFT'
        self._xcfun = 'B3LYP'
        self._basis = '6-31+G*'
        self._use_caps = False
        for key in kwargs.keys():
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            else:
                # TODO: replace exit with exception
                exit('ERROR: unknown region property "{0}"'.format(key))

    @property
    def fragments(self):
        return self._fragments

    @fragments.setter
    def fragments(self, fragments):
        self._fragments = fragments

    @property
    def program(self):
        return self._program

    @program.setter
    def program(self, program):
        assert isinstance(program, str)
        self._program = program.lower()

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, method):
        assert isinstance(method, str)
        self._method = method

    @property
    def xcfun(self):
        return self._xcfun

    @xcfun.setter
    def xcfun(self, xcfun):
        assert isinstance(xcfun, str)
        self._xcfun = xcfun

    @property
    def basis(self):
        return self._basis

    @basis.setter
    def basis(self, basis):
        if isinstance(basis, list):
            assert all(isinstance(bas, str) for bas in basis)
            if len(basis) != self.fragments.number_of_atoms:
                # TODO use exception
                exit('ERROR: either specify one basis set or a list corresponding to number of '
                     'atoms in fragment')
        else:
            assert isinstance(basis, str)
        self._basis = basis

    @property
    def use_caps(self):
        return self._use_caps

    @use_caps.setter
    def use_caps(self, use_caps):
        assert isinstance(use_caps, bool)
        self._use_caps = use_caps

    def write_xyz(self, filename=None):
        if filename is None:
            filename = 'core_region'
        elements = [atom.element for fragment in self.fragments.values() for atom in fragment.atoms]
        coordinates = [atom.coordinate for fragment in self.fragments.values() for atom in fragment.atoms]
        total_charge = sum([fragment.charge for fragment in self.fragments.values()])
        InputWriters.xyz(elements, coordinates, total_charge, filename)

#    def fit_caps(self):


class RegionDict(dict):
    """Region dict methods"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# TODO add model where molecular properties are used, i.e. "coarse-graining"
class Region(object):
    """Container for region attributes and methods"""
    def __init__(self, name, fragments, **kwargs):
        # TODO define standard region specs
        self._name = name
        self._fragments = fragments
        self._use_multipoles = False
        self._use_polarizabilities = False
        self._use_fragment_densities = False
        self._use_exchange_repulsion = False
        self._use_standard_potentials = False
        self._use_lennard_jones = False
        self._use_mfcc = False
        # general options
        self._program = 'dalton'
        self._method = 'DFT'
        self._xcfun = 'B3LYP'
        self._basis = 'loprop-6-31+G*'
        # multipoles options
        self._multipole_order = 2
        # polarizability options
        self._polarizability_order = (1, 1)
        self._isotropic_polarizabilities = False
        # fragment_densities options
        # exchange_repulsion options
        self._exchange_repulsion_factor = 0.8
        # standard potentials options
        self._standard_potential_model = 'sep'
        self._standard_potential_exclusion_type = 'fragment'
        # lennard jones options
        # mfcc options
        self._mfcc_order = 2
        self._mfcc_fragments = FragmentDict()
        for key in kwargs.keys():
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            else:
                # TODO: replace exit with exception
                exit('ERROR: unknown region property "{0}"'.format(key))

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        assert isinstance(name, str)
        self._name = name

    @property
    def fragments(self):
        return self._fragments

    @fragments.setter
    def fragments(self, fragments):
        self._fragments = fragments

    @property
    def use_multipoles(self):
        return self._use_multipoles

    @use_multipoles.setter
    def use_multipoles(self, multipoles):
        assert isinstance(multipoles, bool)
        self._use_multipoles = multipoles

    @property
    def multipole_order(self):
        return self._multipole_order

    @multipole_order.setter
    def multipole_order(self, order):
        assert isinstance(order, int)
        self._multipole_order = order

    @property
    def program(self):
        return self._program

    @program.setter
    def program(self, program):
        assert isinstance(program, str)
        self._program = program.lower()

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, method):
        assert isinstance(method, str)
        self._method = method

    @property
    def xcfun(self):
        return self._xcfun

    @xcfun.setter
    def xcfun(self, xcfun):
        assert isinstance(xcfun, str)
        self._xcfun = xcfun

    @property
    def basis(self):
        return self._basis

    @basis.setter
    def basis(self, basis):
        assert isinstance(basis, str)
        self._basis = basis

    @property
    def use_polarizabilities(self):
        return self._use_polarizabilities

    @use_polarizabilities.setter
    def use_polarizabilities(self, use_polarizabilities):
        assert isinstance(use_polarizabilities, bool)
        self._use_polarizabilities = use_polarizabilities

    @property
    def polarizability_order(self):
        return self._polarizability_order

    @polarizability_order.setter
    def polarizability_order(self, order):
        assert all(isinstance(value, int) for value in order)
        self._polarizability_order = order

    @property
    def isotropic_polarizabilities(self):
        return self._isotropic_polarizabilities

    @isotropic_polarizabilities.setter
    def isotropic_polarizabilities(self, isotropic_polarizabilities):
        assert isinstance(isotropic_polarizabilities, bool)
        self._isotropic_polarizabilities = isotropic_polarizabilities

    @property
    def use_fragment_densities(self):
        return self._use_fragment_densities

    @use_fragment_densities.setter
    def use_fragment_densities(self, use_fragments_densities):
        assert isinstance(use_fragments_densities, bool)
        self._use_fragment_densities = use_fragments_densities

    @property
    def use_exchange_repulsion(self):
        return self._use_exchange_repulsion

    @use_exchange_repulsion.setter
    def use_exchange_repulsion(self, use_exchange_repulsion):
        assert isinstance(use_exchange_repulsion, bool)
        self._use_exchange_repulsion = use_exchange_repulsion

    @property
    def exchange_repulsion_factor(self):
        return self._exchange_repulsion_factor

    @exchange_repulsion_factor.setter
    def exchange_repulsion_factor(self, exchange_repulsion_factor):
        assert isinstance(exchange_repulsion_factor, float)
        self._exchange_repulsion_factor = exchange_repulsion_factor

    @property
    def use_standard_potentials(self):
        return self._use_standard_potentials

    @use_standard_potentials.setter
    def use_standard_potentials(self, use_standard_potentials):
        assert isinstance(use_standard_potentials, bool)
        self._use_standard_potentials = use_standard_potentials

    @property
    def standard_potential_model(self):
        return self._standard_potential_model

    @standard_potential_model.setter
    def standard_potential_model(self, standard_potential_model):
        assert isinstance(standard_potential_model, str)
        self._standard_potential_model = standard_potential_model.lower()

    @property
    def standard_potential_exclusion_type(self):
        return self._standard_potential_exclusion_type

    @standard_potential_exclusion_type.setter
    def standard_potential_exclusion_type(self, standard_potential_exclusion_type):
        assert isinstance(standard_potential_exclusion_type, str)
        self._standard_potential_exclusion_type = standard_potential_exclusion_type.lower()

    @property
    def use_lennard_jones(self):
        return self._use_lennard_jones

    @use_lennard_jones.setter
    def use_lennard_jones(self, use_lennard_jones):
        assert isinstance(use_lennard_jones, bool)
        self._use_lennard_jones = use_lennard_jones

    @property
    def use_mfcc(self):
        return self._use_mfcc

    @use_mfcc.setter
    def use_mfcc(self, use_mfcc):
        assert isinstance(use_mfcc, bool)
        self._use_mfcc = use_mfcc

    @property
    def mfcc_order(self):
        return self._mfcc_order

    @mfcc_order.setter
    def mfcc_order(self, mfcc_order):
        assert isinstance(mfcc_order, int)
        self._mfcc_order = mfcc_order

    @property
    def mfcc_fragments(self):
        return self._mfcc_fragments

    # def consistency_check(self):
    #     print('WARNING: consistency check not implement yet')
    #     pass

    def create_mfcc_fragments(self, bond_threshold=1.15):
        assert isinstance(bond_threshold, float)
        for fragment in self.fragments.values():
            fragment.create_mfcc_fragments(self.mfcc_order, bond_threshold)
            self._mfcc_fragments[fragment.capped_fragment.identifier] = fragment.capped_fragment
            for concap in list(fragment.concaps.values()):
                identifiers = concap.identifier.split('-')
                if not all(identifier in self.fragments for identifier in identifiers):
                    fragment.concaps.pop(concap.identifier)
                else:
                    self._mfcc_fragments[concap.identifier] = concap

    def write_xyz(self, filename=None):
        if filename is None:
            filename = self.name
        elements = [atom.element for fragment in self.fragments.values() for atom in fragment.atoms]
        coordinates = [atom.coordinate for fragment in self.fragments.values() for atom in fragment.atoms]
        charge = sum([fragment.charge for fragment in self.fragments.values()])
        InputWriters.xyz(elements, coordinates, charge, filename)
