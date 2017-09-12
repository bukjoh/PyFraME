# coding: utf-8
"""Blablabla"""

import collections

import numpy as np

from .atoms import AtomList, Atom
from .writers import InputWriters
from .utils import compute_distance_matrix, get_bond_length, scale_bond_length, compute_distance

__all__ = ['FragmentDict', 'Fragment', 'find_nearest_atom', 'find_bonded_fragments',
           'find_bonded_atoms', 'find_bonded_heavy_atoms', 'find_bonded_hydrogens']

WATER_NAMES = ['SOL', 'HOH', 'H2O', 'WAT', 'T3P', 'T3H', 'T4P', 'T4E', 'T5P', 'TIP3', 'SPC']


class FragmentDict(collections.OrderedDict):

    """Fragment dictionary methods"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __add__(self, other):
        new_fragments = FragmentDict()
        for fragment in self.values():
            new_fragments[fragment.identifier] = fragment
        for fragment in other.values():
            new_fragments[fragment.identifier] = fragment
        return new_fragments

    def __radd__(self, other):
        if isinstance(other, int):
            return self
        else:
            return self.__add__(other)

    def write_xyz(self, filename=None):
        combined_fragments = sum(self.values())
        if filename is None:
            filename = combined_fragments.identifier
        combined_fragments.write_xyz(filename)


class Fragment(object):

    """Container for fragment attributes and methods"""

    def __init__(self, **kwargs):
        self.name = ''
        self.number = None
        self.chain_id = ''
        self.identifier = ''
        # self._spin_multiplicity = None
        self.atoms = AtomList()
        self.bonded_fragments = []
        self.capped_fragment = None
        self.concaps = FragmentDict()
        self.region = None
        for key in kwargs.keys():
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            else:
                # TODO: replace exit with exception
                exit('ERROR: unknown fragment property "{0}"'.format(key))

    def __str__(self):
        msg = 'fragment name={0},'.format(self.name)
        msg += ' number={0},'.format(self.number)
        msg += ' chain_id={0}\n'.format(self.chain_id)
        msg += '{0}\n'.format('-' * len(msg))
        for atom in self.atoms:
            msg += atom.__str__() + '\n'
        return msg

    def __add__(self, other):
        new_fragment = self.copy()
        # new_fragment.identifier += '-{0}'.format(other.identifier)
        # TODO handle spin multiplicity
        # if self.spin_multiplicity is not None or other.spin_multiplicity is not None:
        #     exit('ERROR: cannot handle spin multiplicity')
        for atom in other.atoms:
            new_fragment.atoms.append(atom.copy())
        new_fragment.bonded_fragments = []
        for bonded_fragment in self.bonded_fragments:
            if bonded_fragment.identifier == other.identifier:
                continue
            overlap = False
            for bonded_atom in bonded_fragment.atoms:
                if bonded_atom.number in other.atoms:
                    overlap = True
                    break
            if overlap:
                continue
            new_fragment.bonded_fragments.append(bonded_fragment)
        for bonded_fragment in other.bonded_fragments:
            if bonded_fragment.identifier == self.identifier:
                continue
            overlap = False
            for bonded_atom in bonded_fragment.atoms:
                if bonded_atom.number in self.atoms:
                    overlap = True
                    break
            if overlap:
                continue
            new_fragment.bonded_fragments.append(bonded_fragment)
        return new_fragment

    def __radd__(self, other):
        if isinstance(other, int):
            return self.copy()
        else:
            return self.__add__(other)

    @property
    def charge(self):
        charge = 0.0
        for atom in self.atoms:
            charge += atom.charge
        return round(charge)

    @property
    def number_of_atoms(self):
        return len(self.atoms)

    @property
    def center_of_mass(self):
        center_of_mass = np.zeros(3)
        total_mass = 0.0
        for atom in self.atoms:
            total_mass += atom.mass
            center_of_mass += atom.mass * atom.coordinate
        center_of_mass /= total_mass
        return center_of_mass

    @property
    def coordinate_matrix(self):
        return np.array([atom.coordinate for atom in self.atoms])

    @property
    def heavy_coordinate_matrix(self):
        return np.array([atom.coordinate for atom in self.atoms if atom.element != 'H'])

    def add_hydrogen_links(self, bond_threshold=1.2):
        for donor in self.bonded_fragments:
            bonded_atoms = find_bonded_atoms(self, donor, bond_threshold)
            for acceptor_atom, donor_atom in bonded_atoms:
                if 'link' in donor_atom.name:
                    continue
                donor_atom.name += 'link'
                if donor_atom.element != 'H':
                    self.atoms.append(convert2hydrogen(acceptor_atom, donor_atom))
                else:
                    self.atoms.append(donor.atoms.pop(donor_atom.number))

    def add_cap_links(self, bond_threshold=1.2):
        for donor in self.bonded_fragments:
            bonded_atoms = find_bonded_atoms(self, donor, bond_threshold)
            for acceptor_atom, donor_atom in bonded_atoms:
                if 'link' in donor_atom.name:
                    continue
                donor_atom.name += 'cap'
                self.atoms.append(donor.atoms.pop(donor_atom.number))

    def create_mfcc_fragments(self, order=2, bond_threshold=1.2):
        assert isinstance(order, int)
        assert order >= 0
        assert isinstance(bond_threshold, float)
        capped_fragment = self.copy()
        for donor in self.bonded_fragments:
            donor_copy = donor.copy()
            if donor_copy.region == 'core_region':
                # cap with hydrogens
                bonded_atoms = find_bonded_atoms(capped_fragment, donor_copy, bond_threshold)
                for acceptor_atom, donor_atom in bonded_atoms:
                    if donor_atom.element != 'H':
                        link_atom = convert2hydrogen(acceptor_atom, donor_atom)
                        capped_fragment.atoms.append(link_atom)
                    else:
                        capped_fragment.atoms.append(donor_atom.copy())
                continue
            identifiers = sorted([capped_fragment.identifier, donor_copy.identifier])
            identifier = '{0[0]}-{0[1]}'.format(identifiers)
            if identifier in self.concaps:
                concap = None
            else:
                concap = Fragment(identifier=identifier)
            # TODO handle spin multiplicity
            # add heavy atoms from neighbouring fragments corresponding to order
            n = order
            while n > 0:
                bonded_atoms = find_bonded_heavy_atoms(capped_fragment, donor_copy, bond_threshold)
                skip = False
                for acceptor_atom, donor_atom in bonded_atoms:
                    # if donor_atom is directly bonded to donors neighbour then skip
                    for fragment in donor_copy.bonded_fragments:
                        if fragment.identifier == self.identifier:
                            continue
                        for atom in fragment.atoms:
                            r = np.linalg.norm(atom.coordinate - donor_atom.coordinate)
                            if r <= bond_threshold * get_bond_length(atom.element, donor_atom.element):
                                skip = True
                                break
                        if skip:
                            break
                    if skip:
                        skip = False
                        continue
                    atom_copy = donor_copy.atoms.pop(donor_atom.number)
                    capped_fragment.atoms.append(atom_copy)
                    if concap:
                        concap.atoms.append(atom_copy)
                n -= 1
            # TODO expand to donor's neighbours?
            # check for lone atoms in donor and include those as well
            bonded_atoms = find_bonded_atoms(capped_fragment, donor_copy, bond_threshold)
            for acceptor_atom, donor_atom in bonded_atoms:
                lone_atom = True
                for atom in donor_copy.atoms:
                    if atom == donor_atom:
                        continue
                    r = np.linalg.norm(atom.coordinate - donor_atom.coordinate)
                    if r <= bond_threshold * get_bond_length(atom.element, donor_atom.element):
                        lone_atom = False
                        break
                if lone_atom:
                    atom_copy = donor_copy.atoms.pop(donor_atom.number)
                    capped_fragment.atoms.append(atom_copy)
                    if concap:
                        concap.atoms.append(atom_copy)
            # cap with hydrogens
            bonded_atoms = find_bonded_atoms(capped_fragment, donor_copy, bond_threshold)
            for acceptor_atom, donor_atom in bonded_atoms:
                if donor_atom.element != 'H':
                    link_atom = convert2hydrogen(acceptor_atom, donor_atom)
                    capped_fragment.atoms.append(link_atom)
                    if concap:
                        concap.atoms.append(link_atom)
                else:
                    capped_fragment.atoms.append(donor_atom.copy())
                    if concap:
                        concap.atoms.append(donor_atom.copy())
            if concap:
                # now add atoms to concap from the fragment itself
                fragment_copy = self.copy()
                donor_copy = donor.copy()
                n = order
                while n > 0:
                    bonded_atoms = find_bonded_heavy_atoms(donor_copy, fragment_copy, bond_threshold)
                    skip = False
                    for acceptor_atom, donor_atom in bonded_atoms:
                        # if donor_atom is directly bonded to donors neighbour then skip
                        for fragment in fragment_copy.bonded_fragments:
                            if fragment.identifier == donor_copy.identifier:
                                continue
                            for atom in fragment.atoms:
                                r = np.linalg.norm(atom.coordinate - donor_atom.coordinate)
                                if r <= bond_threshold * get_bond_length(atom.element, donor_atom.element):
                                    skip = True
                                    break
                            if skip:
                                break
                        if skip:
                            skip = False
                            continue
                        atom_copy = fragment_copy.atoms.pop(donor_atom.number)
                        concap.atoms.append(atom_copy)
                        donor_copy.atoms.append(atom_copy)
                    n -= 1
                # also include lone atoms here
                bonded_atoms = find_bonded_atoms(donor_copy, fragment_copy, bond_threshold)
                for acceptor_atom, donor_atom in bonded_atoms:
                    lone_atom = True
                    for atom in donor_copy.atoms:
                        if atom == donor_atom:
                            continue
                        r = np.linalg.norm(atom.coordinate - donor_atom.coordinate)
                        if r <= bond_threshold * get_bond_length(atom.element, donor_atom.element):
                            lone_atom = False
                            break
                    if lone_atom:
                        atom_copy = fragment_copy.atoms.pop(donor_atom.number)
                        concap.atoms.append(atom_copy)
                        donor_copy.atoms.append(atom_copy)
                # and we also need hydrogen caps
                bonded_atoms = find_bonded_atoms(donor_copy, fragment_copy, bond_threshold)
                for acceptor_atom, donor_atom in bonded_atoms:
                    if donor_atom.element != 'H':
                        link_atom = convert2hydrogen(acceptor_atom, donor_atom)
                        concap.atoms.append(link_atom)
                    else:
                        concap.atoms.append(donor_atom.copy())
                self.concaps[identifier] = concap
                donor.concaps[identifier] = concap
        self.capped_fragment = capped_fragment

    def copy(self):
        fragment_copy = Fragment()
        fragment_copy.name = self.name
        fragment_copy.number = self.number
        fragment_copy.chain_id = self.chain_id
        fragment_copy.identifier = self.identifier
        # fragment_copy.spin_multiplicity = self.spin_multiplicity
        if self.atoms:
            for atom in self.atoms:
                fragment_copy.atoms.append(atom.copy())
        if self.bonded_fragments:
            fragment_copy.bonded_fragments = []
            for bonded_fragment in self.bonded_fragments:
                fragment_copy.bonded_fragments.append(bonded_fragment)
        fragment_copy.region = self.region
        return fragment_copy

    def write_xyz(self, filename=None):
        if filename is None:
            filename = self.identifier
        elements = [atom.element for atom in self.atoms]
        coordinates = [atom.coordinate for atom in self.atoms]
        InputWriters.xyz(elements, coordinates, self.charge, filename)


def uniquify(fragments):

    unique = []
    unique_fragments = []
    for fragment in fragments:
        if fragment.identifier in unique:
            continue
        unique.append(fragment.identifier)
        unique_fragments.append(fragment)
    return unique_fragments


def convert2hydrogen(acceptor_atom, donor_atom):

    hydrogen = donor_atom.copy()
    hydrogen.element = 'H'
    hydrogen.charge = 0.0
    hydrogen.name += 'link'
    hydrogen.coordinate = np.array(scale_bond_length(acceptor_atom, hydrogen))
    return hydrogen


def find_bonded_fragments(acceptor, donors, bond_threshold=1.2):
    """Find all fragments that are bonded to acceptor fragment"""
    acceptor_coordinate_matrix = acceptor.heavy_coordinate_matrix
    acceptor_atoms = [atom for atom in acceptor.atoms if atom.element != 'H']
    bonded_fragments = []
    if acceptor.name in WATER_NAMES:
        return bonded_fragments
    for donor in donors.values():
        if acceptor.identifier == donor.identifier:
            continue
        if donor.name in WATER_NAMES:
            continue
        distances = compute_distance_matrix(acceptor_coordinate_matrix, donor.heavy_coordinate_matrix)
        first, second = np.unravel_index(distances.argmin(), distances.shape)
        distance = distances[first][second]
        donor_atoms = [atom for atom in donor.atoms if atom.element != 'H']
        acceptor_atom = acceptor_atoms[first]
        donor_atom = donor_atoms[second]
        # TODO consistency checks
        if distance < 0.5:
            # TODO: replace exit with exception
            msg = 'ERROR: very short distance detected ({0} Å)'.format(distance)
            msg += 'between atoms {0} and {1}'.format(acceptor_atom.element, donor_atom.element)
            msg += ' in fragments {0} and {1}'.format(acceptor.identifier, donor.identifier)
            exit(msg)
        if distance <= bond_threshold * get_bond_length(acceptor_atom.element, donor_atom.element):
            bonded_fragments.append(donor)
    return bonded_fragments


def find_nearest_atom(atom, fragment):
    distances = compute_distance_matrix(np.array(atom.coordinate, ndmin=2), fragment.coordinate_matrix)
    index = distances.argmin()
    return fragment.atoms[index]


def find_bonded_atoms(acceptor, donor, bond_threshold=1.2):
    bonded_atoms = []
    if acceptor.number_of_atoms == 0 or donor.number_of_atoms == 0:
        return bonded_atoms
    distances = compute_distance_matrix(acceptor.coordinate_matrix, donor.coordinate_matrix)
    for i, row in enumerate(distances):
        for j, r in enumerate(row):
            if r > 5.4:
                continue
            acceptor_atom = acceptor.atoms[i]
            donor_atom = donor.atoms[j]
            if r <= bond_threshold * get_bond_length(acceptor_atom.element, donor_atom.element):
                bonded_atoms.append((acceptor_atom, donor_atom))
    return bonded_atoms


def find_bonded_heavy_atoms(acceptor, donor, bond_threshold=1.2):
    bonded_heavy_atoms = []
    if acceptor.number_of_atoms == 0 or donor.number_of_atoms == 0:
        return bonded_heavy_atoms
    distances = compute_distance_matrix(acceptor.coordinate_matrix, donor.coordinate_matrix)
    for i, row in enumerate(distances):
        for j, r in enumerate(row):
            if r > 5.4:
                continue
            donor_atom = donor.atoms[j]
            if donor_atom.element == 'H':
                continue
            acceptor_atom = acceptor.atoms[i]
            if r <= bond_threshold * get_bond_length(acceptor_atom.element, donor_atom.element):
                bonded_heavy_atoms.append((acceptor_atom, donor_atom))
    return bonded_heavy_atoms


def find_bonded_hydrogens(acceptor, donor, bond_threshold=1.2):
    bonded_hydrogens = []
    if acceptor.number_of_atoms == 0 or donor.number_of_atoms == 0:
        return bonded_hydrogens
    distances = compute_distance_matrix(acceptor.coordinate_matrix, donor.coordinate_matrix)
    for i, row in enumerate(distances):
        for j, r in enumerate(row):
            if r > 3.0:
                continue
            donor_atom = donor.atoms[j]
            if donor_atom.element != 'H':
                continue
            acceptor_atom = acceptor.atoms[i]
            if r <= bond_threshold * get_bond_length(acceptor_atom.element, donor_atom.element):
                bonded_hydrogens.append((acceptor_atom, donor_atom))
    return bonded_hydrogens


# def detect_hydrogen_bonds(polymer, polymers, all_frags):
#
#     '''Detect backbone hydrogen bonded fragments'''
#
#     # TODO: expand to other hydrogen bonds, e.g. side chain?
#     # Hydrogen bond rules:
#     # N-H--O=C where
#     # 1: H-O < 2.5Aa
#     # 2: N-O < 3.9Aa
#     # 3. N-H-O > 90.0
#     # 4. N-O-C > 90.0
#     # 5. H-O-C > 90.0
#     for frag in polymer.fragments:
#         hb_bonds = []
#         if frag.name == 'PRO':
#             continue
#         Hf = frag.names.index('H')
#         Nf = frag.names.index('N')
#         Cf = frag.names.index('C')
#         Of = frag.names.index('O')
#         for poly in polymers:
#             for temp in polymer.fragments:
#                 if temp.seqid == frag.seqid:
#                     continue
#                 Hd = temp.names.index('H')
#                 Nd = temp.names.index('N')
#                 Cd = temp.names.index('C')
#                 Od = temp.names.index('O')
#                 r_HO = distance(frag.coords[Hf], temp.coords[Od])
#                 if r_HO < 2.5:
#                     r_NO = distance(frag.coords[Nf], temp.coords[Od])
#                     if r_NO < 3.9:
#                         a_NHO = angle(frag.coords[Nf], frag.coords[Hf],
#                                       temp.coords[Od])
#                         if a_NHO > (np.pi * 0.5):
#                             a_NOC = angle(frag.coords[Nf],
#                                           temp.coords[Od],
#                                           temp.coords[Cd])
#                             if a_NOC > (np.pi * 0.5):
#                                 a_HOC = angle(frag.coords[Hf],
#                                               temp.coords[Od],
#                                               temp.coords[Cd])
#                                 if a_HOC > (np.pi * 0.5):
#                                     hb_bonds.append(temp)
#                 r_OH = distance(frag.coords[Of], temp.coords[Hd])
#                 if r_OH < 2.5:
#                     r_ON = distance(frag.coords[Of], temp.coords[Nd])
#                     if r_ON < 3.9:
#                         a_OHN = angle(frag.coords[Of], temp.coords[Hd],
#                                       temp.coords[Nd])
#                         if a_OHN > (np.pi * 0.5):
#                             a_CON = angle(frag.coords[Cf],
#                                           frag.coords[Of],
#                                           temp.coords[Nd])
#                             if a_CON > (np.pi * 0.5):
#                                 a_COH = angle(frag.coords[Cf],
#                                               frag.coords[Of],
#                                               temp.coords[Nd])
#                                 if a_COH > (np.pi * 0.5):
#                                     hb_bonds.append(temp)
#
#         frag.hb_bonds = hb_bonds
