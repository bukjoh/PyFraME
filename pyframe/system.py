# coding=utf-8
""" The module 'system' contains the class 'MolecularSystem', which defines the moleculer system."""

import os.path

from .atoms import AtomList
from .fragments import FragmentDict, find_bonded_fragments
from .regions import CoreRegion, RegionDict, Region
from .potentials import PotentialDict
from .readers import InputReaders, read_input_file
from .utils import compute_distance, get_minimum_distance
from .writers import InputWriters


__all__ = ['MolecularSystem']


class MolecularSystem(object):
    """Define a molecular system."""

    def __init__(self, input_file, name=None, input_reader=None, bond_threshold=None):
        assert isinstance(input_file, str), 'Input file must be given as a string'
        assert os.path.isfile(input_file)
        self._fragments = FragmentDict()
        self._potential = PotentialDict()
        self._core_region = None
        self._regions = RegionDict()
        self._name = None
        self._bond_threshold = 1.2
        file_path, filename = os.path.split(input_file)
        filename, file_ext = os.path.splitext(filename)
        if name:
            self._name = name
        else:
            self._name = filename
        if bond_threshold:
            self._bond_threshold = bond_threshold
        if input_reader:
            self._fragments = read_input_file(input_file, input_reader)
        else:
            reader = 'read_{0}'.format(file_ext.strip('.'))
            if hasattr(InputReaders, reader):
                self._fragments = read_input_file(input_file, getattr(InputReaders, reader))
            else:
                # TODO replace with exception
                exit('ERROR: no input reader found for this input format')
        if not self.fragments:
            # TODO replace with exception
            exit('ERROR: no fragments defined')
        self.analyse_bonding(self.fragments)
        self._fragments_backup = FragmentDict(self.fragments)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        assert isinstance(name, str)
        self._name = name

    @property
    def bond_threshold(self):
        return self._bond_threshold

    @bond_threshold.setter
    def bond_threshold(self, bond_threshold):
        assert isinstance(bond_threshold, float)
        self._bond_threshold = bond_threshold

    @property
    def fragments(self):
        return self._fragments

    @property
    def potential(self):
        return self._potential

    @property
    def core_region(self):
        return self._core_region

    @property
    def regions(self):
        return self._regions

    def analyse_bonding(self, fragments=None):
        if fragments is None:
            fragments = self._fragments
        for fragment in fragments.values():
            fragment.bonded_fragments = find_bonded_fragments(fragment, fragments, self.bond_threshold)

    def split_fragment_by_identifier(self, identifier, new_names, fragment_definitions):
        """Split a fragment with the specified identifier into several fragments with different names.

        The identifier must be a string. 'new_names' must be a list of strings.
        'fragment_definitions' must be a list with the same number of items as 'new_names'.
        Each item in the 'fragment_definitions' list must be a list of strings representing atom names.
        
        Fragments with the specified identifier are split based on the specified fragment definitions.

        """
        assert isinstance(identifier, str)
        assert isinstance(new_names, list)
        assert all(isinstance(name, str) for name in new_names)
        assert isinstance(fragment_definitions, list)
        assert all(isinstance(definition, list) for definition in fragment_definitions)
        assert all(isinstance(atom_name, str) for definition in fragment_definitions for atom_name in definition)
        assert len(new_names) == len(fragment_definitions)
        original_fragment = self.fragments.pop(identifier)
        new_identifiers = []
        for new_name, definition in zip(new_names, fragment_definitions):
            new_fragment = original_fragment.copy()
            new_atoms = AtomList()
            for atom in list(original_fragment.atoms):
                if atom.name in definition:
                    new_atoms.append(original_fragment.atoms.pop(atom.number))
            new_fragment.atoms = new_atoms
            new_fragment.name = new_name
            if new_fragment.chain_id:
                new_fragment.identifier = '{0}_{1}_{2}'.format(new_fragment.number, new_fragment.chain_id,
                                                               new_fragment.name)
            else:
                new_fragment.identifier = '{0}_{1}'.format(new_fragment.number, new_fragment.name)
            self.fragments[new_fragment.identifier] = new_fragment
            new_identifiers.append(new_fragment.identifier)
        if original_fragment.number_of_atoms != 0:
            # TODO replace with exception
            exit('ERROR: not all atoms are used when splitting fragment: {}'.format(identifier))
        # re-analyze bonding after splitting
        fragments = FragmentDict()
        fragments_and_neighbours = FragmentDict()
        for bonded_fragment in original_fragment.bonded_fragments:
            fragments[bonded_fragment.identifier] = bonded_fragment
            fragments_and_neighbours[bonded_fragment.identifier] = bonded_fragment
            for neighbour_fragment in bonded_fragment.bonded_fragments:
                if neighbour_fragment.identifier == identifier:
                    continue
                fragments_and_neighbours[neighbour_fragment.identifier] = neighbour_fragment
        for new_identifier in new_identifiers:
            fragments[new_identifier] = self.fragments[new_identifier]
            fragments_and_neighbours[new_identifier] = self.fragments[new_identifier]
        for fragment in fragments.values():
            fragment.bonded_fragments = find_bonded_fragments(fragment, fragments_and_neighbours, self.bond_threshold)

    def split_fragment_by_name(self, name, new_names, fragment_definitions):
        """Split a fragment with the specified name into several fragments with different names.

        The value for 'name' must be a string. 'new_names' must be a list of strings.
        'fragment_definitions' must be a list with the same number of items as 'new_names'.
        Each item in the 'fragment_definitions' list must be a list of strings representing atom names.
        
        Fragments with the specified 'name' are split based on the specified fragment definitions.

        """
        assert isinstance(name, str)
        assert isinstance(new_names, list)
        assert all(isinstance(new_name, str) for new_name in new_names)
        assert isinstance(fragment_definitions, list)
        assert all(isinstance(definition, list) for definition in fragment_definitions)
        assert all(isinstance(atom_name, str) for definition in fragment_definitions for atom_name in definition)
        assert len(new_names) == len(fragment_definitions)
        identifiers = []
        for fragment in self.fragments.values():
            if fragment.name == name:
                identifiers.append(fragment.identifier)
        for identifier in identifiers:
            original_fragment = self.fragments.pop(identifier)
            new_identifiers = []
            for new_name, definition in zip(new_names, fragment_definitions):
                new_fragment = original_fragment.copy()
                new_atoms = AtomList()
                for atom in list(original_fragment.atoms):
                    if atom.name in definition:
                        new_atoms.append(original_fragment.atoms.pop(atom.number))
                new_fragment.atoms = new_atoms
                new_fragment.name = new_name
                if new_fragment.chain_id:
                    new_fragment.identifier = '{0}_{1}_{2}'.format(new_fragment.number, new_fragment.chain_id,
                                                                   new_fragment.name)
                else:
                    new_fragment.identifier = '{0}_{1}'.format(new_fragment.number, new_fragment.name)
                self.fragments[new_fragment.identifier] = new_fragment
                new_identifiers.append(new_fragment.identifier)
            if original_fragment.number_of_atoms != 0:
                # TODO replace with exception
                exit('ERROR: not all atoms are used when splitting fragment: {}'.format(identifier))
            # re-analyze bonding after splitting
            fragments = FragmentDict()
            fragments_and_neighbours = FragmentDict()
            for bonded_fragment in original_fragment.bonded_fragments:
                fragments[bonded_fragment.identifier] = bonded_fragment
                fragments_and_neighbours[bonded_fragment.identifier] = bonded_fragment
                for neighbour_fragment in bonded_fragment.bonded_fragments:
                    if neighbour_fragment.identifier == identifier:
                        continue
                    fragments_and_neighbours[neighbour_fragment.identifier] = neighbour_fragment
            for new_identifier in new_identifiers:
                fragments[new_identifier] = self.fragments[new_identifier]
                fragments_and_neighbours[new_identifier] = self.fragments[new_identifier]
            for fragment in fragments.values():
                fragment.bonded_fragments = find_bonded_fragments(fragment, fragments_and_neighbours,
                                                                  self.bond_threshold)

    # def merge_fragments_by_identifier(self, identifiers, new_name):
    #     assert isinstance(identifiers, list)
    #     assert all(isinstance(identifier, str) for identifier in identifiers)
    #     assert isinstance(new_name, str)
    #     fragments = []
    #     for identifier in identifiers:
    #         fragments.append(self.fragments.pop(identifier))
    #     new_fragment = sum(fragments)
    #     new_fragment.name = new_name
    #     if new_fragment.chain_id:
    #         new_fragment.identifier = '{0}_{1}_{2}'.format(new_fragment.number, new_fragment.chain_id,
    #                                                        new_fragment.name)
    #     else:
    #         new_fragment.identifier = '{0}_{1}'.format(new_fragment.number, new_fragment.name)
    #     self.fragments[new_fragment.identifier] = new_fragment

    def get_fragments_by_identifier(self, identifiers):
        """Return a fragment dictionary based on a list of identifiers.

        An identifier is a unique string that identifies a fragment.
        This can for example be a combination of fragment number, chain id and fragment name
        from a PDB file, such as '1_A_SOL'.
 
        """
        assert isinstance(identifiers, list)
        assert all(isinstance(identifier, str) for identifier in identifiers)
        fragments = FragmentDict()
        for identifier in identifiers:
            if identifier in self.fragments:
                fragments[identifier] = self.fragments.pop(identifier)
        return fragments

    def get_fragments_by_name(self, names):
        """Return a fragment dictionary based on a list of fragment names.

        Each fragment name must be a string.

        """
        assert isinstance(names, list)
        assert all(isinstance(name, str) for name in names)
        fragments = FragmentDict()
        for fragment in list(self.fragments.values()):
            if fragment.name in names:
                fragments[fragment.identifier] = self.fragments.pop(fragment.identifier)
        return fragments

    def get_fragments_by_number(self, numbers):
        """Return a fragment dictionary based on a list of fragment numbers.

        Each fragment number must be an integer.

        """
        assert isinstance(numbers, list)
        assert all(isinstance(number, int) for number in numbers)
        fragments = FragmentDict()
        if any('-' in number for number in numbers):
            expanded_numbers = []
            for number in numbers:
                if '-' in number:
                    no_range = number.split('-')
                    for value in range(int(no_range[0]), int(no_range[1]) + 1):
                        expanded_numbers.append(value)
                else:
                    expanded_numbers.append(number)
            numbers = list(set(expanded_numbers))
        for fragment in list(self.fragments.values()):
            if fragment.number in numbers:
                fragments[fragment.identifier] = self.fragments.pop(fragment.identifier)
        return fragments

    def get_fragments_by_chain_id(self, chain_ids):
        """Return a fragment dictionary based on a list of chain ids.

        Each chain id must be a string.

        """
        assert isinstance(chain_ids, list)
        assert all(isinstance(chain_id, str) for chain_id in chain_ids)
        fragments = FragmentDict()
        for fragment in list(self.fragments.values()):
            if fragment.chain_id in chain_ids:
                fragments[fragment.identifier] = self.fragments.pop(fragment.identifier)
        return fragments

    def get_fragments_by_charge(self, charges):
        """Return a fragment dictionary based on a list of charges.

        Each charge must be an integer.
        """
        assert isinstance(charges, list)
        assert all(isinstance(charge, int) for charge in charges)
        fragments = FragmentDict()
        for fragment in list(self.fragments.values()):
            if fragment.charge in charges:
                fragments[fragment.identifier] = self.fragments.pop(fragment.identifier)
        return fragments

    def get_fragments_by_distance(self, distance, reference, use_center_of_mass=True, protect_molecules=True):
        """Return a fragment dictionary based on a distance criterion from a reference fragment.

        The distance must be a floating-point number. The reference must be a fragment.

        The variable 'use_center_of_mass' must be a Boolean and defaults to 'True'. If 'True', the centre of mass
        of both the fragment and reference are used to evaluate the distance. If 'False', the distance between
        individual atoms will be evaluated.

        The variable 'protect_molecules' must be a Boolean and defaults to 'True'. If 'True', all fragments that are
        covalently bonded to the fragment that fulfills the distance criterion will be included.

        """
        assert isinstance(distance, float)
        assert isinstance(reference, FragmentDict)
        assert isinstance(use_center_of_mass, bool)
        assert isinstance(protect_molecules, bool)
        fragments = FragmentDict()
        if use_center_of_mass:
            for fragment in list(self.fragments.values()):
                for reference_fragment in reference.values():
                    if compute_distance(fragment.center_of_mass, reference_fragment.center_of_mass) <= distance:
                        if fragment.identifier in self.fragments:
                            fragments[fragment.identifier] = self.fragments.pop(fragment.identifier)
        else:
            for fragment in list(self.fragments.values()):
                for reference_fragment in reference.values():
                    if get_minimum_distance(fragment, reference_fragment) <= distance:
                        if fragment.identifier in self.fragments:
                            fragments[fragment.identifier] = self.fragments.pop(fragment.identifier)
        if protect_molecules:
            num_fragments = len(fragments)
            while True:
                for fragment in list(fragments.values()):
                    for bonded_fragment in fragment.bonded_fragments:
                        if bonded_fragment.identifier in fragments.keys():
                            continue
                        if bonded_fragment.identifier in self.fragments.keys():
                            fragments[bonded_fragment.identifier] = self.fragments.pop(bonded_fragment.identifier)
                if len(fragments) != num_fragments:
                    num_fragments = len(fragments)
                elif len(fragments) == num_fragments:
                    break
        return fragments

    def get_fragments_by_distance_and_name(self, distance, names, reference, use_center_of_mass=True,
                                           protect_molecules=True):
        """Return a fragment dictionary based on a distance criterion and a list of fragment names.

        The distance must be a floating-point number. The reference must be a fragment dictionary.
        Each fragment name must be a string.

        The variable 'use_center_of_mass' must be a Boolean and defaults to 'True'. If 'True', the centre of mass
        of both the fragment and reference are used to evaluate the distance. If 'False', the distance between
        individual atoms will be evaluated.

        The variable 'protect_molecules' must be a Boolean and defaults to 'True'. If 'True', all fragments that are
        covalently bonded to the fragment that fulfills the distance criterion will be included.

        """
        assert isinstance(distance, float)
        assert isinstance(names, list)
        assert all(isinstance(name, str) for name in names)
        assert isinstance(reference, FragmentDict)
        assert isinstance(use_center_of_mass, bool)
        fragments = FragmentDict()
        if use_center_of_mass:
            for fragment in list(self.fragments.values()):
                if fragment.name not in names:
                    continue
                for reference_fragment in reference.values():
                    if compute_distance(fragment.center_of_mass, reference_fragment.center_of_mass) <= distance:
                        if fragment.identifier in self.fragments:
                            fragments[fragment.identifier] = self.fragments.pop(fragment.identifier)
        else:
            for fragment in list(self.fragments.values()):
                if fragment.name not in names:
                    continue
                for reference_fragment in reference.values():
                    if get_minimum_distance(fragment, reference_fragment) <= distance:
                        if fragment.identifier in self.fragments:
                            fragments[fragment.identifier] = self.fragments.pop(fragment.identifier)
        if protect_molecules:
            num_fragments = len(fragments)
            while True:
                for fragment in list(fragments.values()):
                    for bonded_fragment in fragment.bonded_fragments:
                        if bonded_fragment.identifier in fragments.keys():
                            continue
                        if bonded_fragment.identifier in self.fragments.keys():
                            fragments[bonded_fragment.identifier] = self.fragments.pop(bonded_fragment.identifier)
                if len(fragments) != num_fragments:
                    num_fragments = len(fragments)
                elif len(fragments) == num_fragments:
                    break
        return fragments

    def get_fragments_by_distance_and_chain_id(self, distance, chain_ids, reference, use_center_of_mass=True,
                                               protect_molecules=True):
        """Return a fragment dictionary based on a distance criterion and a list of chain ids.

        The distance must be a floating-point number. The reference must be a fragment dictionary.
        Each chain id must be a string.

        The variable 'use_center_of_mass' must be a Boolean and defaults to 'True'. If 'True', the centre of mass
        of both the fragment and reference are used to evaluate the distance. If 'False', the distance between
        individual atoms will be evaluated.

        The variable 'protect_molecules' must be a Boolean and defaults to 'True'. If 'True', all fragments that are
        covalently bonded to the fragment that fulfills the distance criterion will be included.

        """
        assert isinstance(distance, float)
        assert isinstance(chain_ids, list)
        assert all(isinstance(name, str) for name in chain_ids)
        assert isinstance(reference, FragmentDict)
        assert isinstance(use_center_of_mass, bool)
        fragments = FragmentDict()
        if use_center_of_mass:
            for fragment in list(self.fragments.values()):
                if fragment.chain_id not in chain_ids:
                    continue
                for reference_fragment in reference.values():
                    if compute_distance(fragment.center_of_mass, reference_fragment.center_of_mass) <= distance:
                        if fragment.identifier in self.fragments:
                            fragments[fragment.identifier] = self.fragments.pop(fragment.identifier)
        else:
            for fragment in list(self.fragments.values()):
                if fragment.chain_id not in chain_ids:
                    continue
                for reference_fragment in reference.values():
                    if get_minimum_distance(fragment, reference_fragment) <= distance:
                        if fragment.identifier in self.fragments:
                            fragments[fragment.identifier] = self.fragments.pop(fragment.identifier)
        if protect_molecules:
            num_fragments = len(fragments)
            while True:
                for fragment in list(fragments.values()):
                    for bonded_fragment in fragment.bonded_fragments:
                        if bonded_fragment.identifier in fragments.keys():
                            continue
                        if bonded_fragment.identifier in self.fragments.keys():
                            fragments[bonded_fragment.identifier] = self.fragments.pop(bonded_fragment.identifier)
                if len(fragments) != num_fragments:
                    num_fragments = len(fragments)
                elif len(fragments) == num_fragments:
                    break
        return fragments

    def add_region(self, name, fragments, **kwargs):
        """Add a region to the molecular system."""
        for fragment in fragments.values():
            fragment.region = name
        self._regions[name] = Region(name, fragments, **kwargs)

    def set_core_region(self, fragments, **kwargs):
        """Set the core region of the molucular system.."""
        for fragment in fragments.values():
            fragment.region = 'core_region'
        self._core_region = CoreRegion(fragments, **kwargs)

    def write_core(self, filename=None):
        """Write a Dalton molecule file for the core region of the molecular system.

        The core region must be set before this function can be called.
        The filename will be the variable 'filename' with the extensiion '.mol'.
        The attribute 'filename' is optional and defaults to the name of the molecular system plus '_core'.

        """
        if self._core_region is None:
            # TODO: replace exit with exception
            exit('ERROR: core region is not defined')
        if filename is None:
            filename = self.name
        writer = self.core_region.program + '_core'
        if hasattr(InputWriters, writer):
            getattr(InputWriters, writer)(self, filename)
        else:
            # TODO replace with exception
            exit('ERROR: {0} does not exist'.format(writer))

    def write_potential(self, filename=None):
        """Write the embedding potential for the molecular system to a file.
        
        The filename will be the variable 'filename' with the extension '.pot'.
        The variable 'filename' defaults to the name of the the molecular system with the extension '.pot'.

        """
        if self.potential is None:
            # TODO: replace exit with exception
            exit('ERROR: potential is not defined')
        if filename is None:
            filename = self.name
        InputWriters.pelib_potential(self, filename)
        # InputWriters.frame_potential(self, filename)

    def reset(self):
        """Reset the molecular system.

        The fragment dictionary of the molecular system is reset to its initial value.
        The potential dictionary is emptied and the core region is removed.

        """
        self._fragments = FragmentDict(self._fragments_backup)
        self._potential = PotentialDict()
        self._core_region = None
        self._regions = RegionDict()
