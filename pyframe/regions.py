# coding=utf-8
"""Blablabla"""

import collections

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


class RegionDict(collections.OrderedDict):
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
        self._is_inner_region = False
        self._is_outer_region = False
        self._use_multipoles = False
        self._use_polarizabilities = False
        self._use_fragment_densities = False
        self._use_exchange_repulsion = False
        self._use_standard_potentials = False
        self._use_lennard_jones = False
        self._use_mfcc = False
        # multipoles options
        self._multipole_order = 2
        self._multipole_program = 'dalton'
        self._multipole_model = 'loprop'
        self._multipole_method = 'DFT'
        self._multipole_xcfun = 'B3LYP'
        self._multipole_basis = 'loprop-6-31+G*'
        self._atomic_multipoles = True
        # polarizability options
        self._polarizability_order = (1, 1)
        self._polarizability_program = 'dalton'
        self._polarizability_model = 'loprop'
        self._polarizability_method = 'DFT'
        self._polarizability_xcfun = 'B3LYP'
        self._polarizability_basis = 'loprop-6-31+G*'
        self._atomic_polarizabilities = True
        self._isotropic_polarizabilities = False
        # fragment_densities options
        self._fragment_density_program = 'dalton'
        # self._fragment_density_model = 'fd'
        self._fragment_density_method = 'DFT'
        self._fragment_density_xcfun = 'B3LYP'
        self._fragment_density_basis = '6-31+G*'
        # exchange_repulsion options
        self._exchange_repulsion_program = 'dalton'
        # self._exchange_repulsion_model = 'hc'
        self._exchange_repulsion_method = 'DFT'
        self._exchange_repulsion_xcfun = 'B3LYP'
        self._exchange_repulsion_basis = '6-31+G*'
        self._exchange_repulsion_factors = [0.8, 0.0, 0.0]
        # standard potentials options
        self._standard_potential_model = 'ff94'
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
        if self.is_inner_region:
            if self.is_outer_region:
                # TODO replace with exception
                exit('ERROR: inner region and outer region are mutually exclusive')
            self.use_fragment_densities = True
            self.use_exchange_repulsion = True
            self.use_polarizabilities = True
        if self.is_outer_region:
            if self.is_inner_region:
                # TODO replace with exception
                exit('ERROR: inner region and outer region are mutually exclusive')
            self.use_multipoles = True
            self.use_polarizabilities = True

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
    def is_inner_region(self):
        return self._is_inner_region

    @is_inner_region.setter
    def is_inner_region(self, is_inner_region):
        assert isinstance(is_inner_region, bool)
        self._is_inner_region = is_inner_region

    @property
    def is_outer_region(self):
        return self._is_outer_region

    @is_outer_region.setter
    def is_outer_region(self, is_outer_region):
        assert isinstance(is_outer_region, bool)
        self._is_outer_region = is_outer_region

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
    def multipole_program(self):
        return self._multipole_program

    @multipole_program.setter
    def multipole_program(self, multipole_program):
        assert isinstance(multipole_program, str)
        self._multipole_program = multipole_program.lower()

    @property
    def multipole_model(self):
        return self._multipole_model

    @multipole_model.setter
    def multipole_model(self, multipole_model):
        assert isinstance(multipole_model, str)
        self._multipole_model = multipole_model.lower()

    @property
    def multipole_method(self):
        return self._multipole_method

    @multipole_method.setter
    def multipole_method(self, method):
        assert isinstance(method, str)
        self._multipole_method = method

    @property
    def multipole_xcfun(self):
        return self._multipole_xcfun

    @multipole_xcfun.setter
    def multipole_xcfun(self, xcfun):
        assert isinstance(xcfun, str)
        self._multipole_xcfun = xcfun

    @property
    def multipole_basis(self):
        return self._multipole_basis

    @multipole_basis.setter
    def multipole_basis(self, basis):
        assert isinstance(basis, str)
        self._multipole_basis = basis

    @property
    def atomic_multipoles(self):
        return self._atomic_multipoles

    @atomic_multipoles.setter
    def atomic_multipoles(self, atomic_multipoles):
        assert isinstance(atomic_multipoles, bool)
        if self.use_mfcc:
            assert atomic_multipoles
        self._atomic_multipoles = atomic_multipoles

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
    def polarizability_program(self):
        return self._polarizability_program

    @polarizability_program.setter
    def polarizability_program(self, polarizability_program):
        assert isinstance(polarizability_program, str)
        self._polarizability_program = polarizability_program.lower()

    @property
    def polarizability_model(self):
        return self._polarizability_model

    @polarizability_model.setter
    def polarizability_model(self, polarizability_model):
        assert isinstance(polarizability_model, str)
        self._polarizability_model = polarizability_model.lower()

    @property
    def polarizability_method(self):
        return self._polarizability_method

    @polarizability_method.setter
    def polarizability_method(self, method):
        assert isinstance(method, str)
        self._polarizability_method = method

    @property
    def polarizability_xcfun(self):
        return self._polarizability_xcfun

    @polarizability_xcfun.setter
    def polarizability_xcfun(self, xcfun):
        assert isinstance(xcfun, str)
        self._polarizability_xcfun = xcfun

    @property
    def polarizability_basis(self):
        return self._polarizability_basis

    @polarizability_basis.setter
    def polarizability_basis(self, basis):
        assert isinstance(basis, str)
        self._polarizability_basis = basis

    @property
    def atomic_polarizabilities(self):
        return self._atomic_polarizabilities

    @atomic_polarizabilities.setter
    def atomic_polarizabilities(self, atomic_polarizabilities):
        assert isinstance(atomic_polarizabilities, bool)
        if self.use_mfcc:
            assert atomic_polarizabilities
        self._atomic_polarizabilities = atomic_polarizabilities

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
    def fragment_density_program(self):
        return self._fragment_density_program

    @fragment_density_program.setter
    def fragment_density_program(self, fragment_density_program):
        assert isinstance(fragment_density_program, str)
        self._fragment_density_program = fragment_density_program.lower()

    # @property
    # def fragment_density_model(self):
    #     return self._fragment_density_model
    #
    # @fragment_density_model.setter
    # def fragment_density_model(self, fragment_density_model):
    #     assert isinstance(fragment_density_model, str)
    #     self._fragment_density_model = fragment_density_model.lower()

    @property
    def fragment_density_method(self):
        return self._fragment_density_method

    @fragment_density_method.setter
    def fragment_density_method(self, method):
        assert isinstance(method, str)
        self._fragment_density_method = method

    @property
    def fragment_density_xcfun(self):
        return self._fragment_density_xcfun

    @fragment_density_xcfun.setter
    def fragment_density_xcfun(self, xcfun):
        assert isinstance(xcfun, str)
        self._fragment_density_xcfun = xcfun

    @property
    def fragment_density_basis(self):
        return self._fragment_density_basis

    @fragment_density_basis.setter
    def fragment_density_basis(self, basis):
        assert isinstance(basis, str)
        self._fragment_density_basis = basis

    @property
    def use_exchange_repulsion(self):
        return self._use_exchange_repulsion

    @use_exchange_repulsion.setter
    def use_exchange_repulsion(self, use_exchange_repulsion):
        assert isinstance(use_exchange_repulsion, bool)
        self._use_exchange_repulsion = use_exchange_repulsion

    @property
    def exchange_repulsion_program(self):
        return self._exchange_repulsion_program

    @exchange_repulsion_program.setter
    def exchange_repulsion_program(self, exchange_repulsion_program):
        assert isinstance(exchange_repulsion_program, str)
        self._exchange_repulsion_program = exchange_repulsion_program

    # @property
    # def exchange_repulsion_model(self):
    #     return self._exchange_repulsion_model
    #
    # @exchange_repulsion_model.setter
    # def exchange_repulsion_model(self, exchange_repulsion_model):
    #     assert isinstance(exchange_repulsion_model, str)
    #     self._exchange_repulsion_model = exchange_repulsion_model

    @property
    def exchange_repulsion_method(self):
        return self._exchange_repulsion_method

    @exchange_repulsion_method.setter
    def exchange_repulsion_method(self, method):
        assert isinstance(method, str)
        self._exchange_repulsion_method = method

    @property
    def exchange_repulsion_xcfun(self):
        return self._exchange_repulsion_xcfun

    @exchange_repulsion_xcfun.setter
    def exchange_repulsion_xcfun(self, xcfun):
        assert isinstance(xcfun, str)
        self._exchange_repulsion_xcfun = xcfun

    @property
    def exchange_repulsion_basis(self):
        return self._exchange_repulsion_basis

    @exchange_repulsion_basis.setter
    def exchange_repulsion_basis(self, basis):
        assert isinstance(basis, str)
        self._exchange_repulsion_basis = basis

    @property
    def exchange_repulsion_factors(self):
        return self._exchange_repulsion_factors

    @exchange_repulsion_factors.setter
    def exchange_repulsion_factors(self, exchange_repulsion_factors):
        assert all(isinstance(factor, float) for factor in exchange_repulsion_factors)
        self._exchange_repulsion_factors = exchange_repulsion_factors

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
        if use_mfcc and self.use_multipoles:
            assert self.atomic_multipoles
        if use_mfcc and self.use_polarizabilities:
            assert self.atomic_polarizabilities
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

    def create_mfcc_fragments(self, bond_threshold=1.2):
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
