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

import os.path
import warnings
import pathlib

import numpy as np
import h5py

from .fragments import FragmentDict, Fragment
from .atoms import AtomList, Atom
from .potentials import PotentialDict, Potential
from .utils import BOHR2AA, AA2BOHR, elements, amino_acid_names


__all__ = ['OutputReaders', 'InputReaders', 'read_standard_potential', 'read_pelib_potential', 'read_input_file',
           'PDBError']


def read_standard_potential(potential_file):

    potential_path = os.path.join(os.path.dirname(__file__), 'data')
    with open('{0}/{1}.csv'.format(potential_path, potential_file), 'r') as pot_file:
        line = pot_file.readline()
        while line:
            if line[0] == '#':
                line = pot_file.readline()
                continue
            break
        props = line.strip().split(',')[2:]
        props = [prop.strip() for prop in props]
        lines = [line.strip().split(',') for line in pot_file.readlines()]
        lines = [[item.strip() for item in line] for line in lines]
        potential = {}
        for line in lines:
            potential[line[0]] = {}
        for line in lines:
            potential[line[0]][line[1]] = {}
        for line in lines:
            for i, values in enumerate(line[2:]):
                try:
                    values = float(values)
                except ValueError:
                    values = [float(value) for value in values.split()]
                except:
                    raise
                potential[line[0]][line[1]][props[i]] = values
    return potential


def read_pelib_potential(filename):
    potential = PotentialDict()
    with open(filename) as input_file:
        line = input_file.readline()
        while line:
            if '@COORDINATES' in line:
                num_sites = int(input_file.readline())
                unit = str(input_file.readline().strip())
                for i in range(num_sites):
                    site = Potential()
                    temp = input_file.readline().split()
                    site.element = temp[0]
                    if unit == 'AU':
                        site.coordinate = [float(value)*BOHR2AA for value in temp[1:4]]
                    elif unit == 'AA':
                        site.coordinate = [float(value) for value in temp[1:4]]
                    else:
                        raise ValueError
                    potential[i+1] = site
            if 'ORDER' in line:
                temp = line.split()
                if len(temp) == 2:
                    multipole_order = int(temp[1])
                    num_multipoles = int(input_file.readline())
                    for i in range(num_multipoles):
                        temp = input_file.readline().split()
                        site_num = int(temp[0])
                        multipole = [float(value) for value in temp[1:]]
                        setattr(potential[site_num], 'M{0}'.format(multipole_order), multipole)
                elif len(temp) == 3:
                    polarizability_order = [int(value) for value in temp[1:]]
                    num_polarizabilities = int(input_file.readline())
                    for i in range(num_polarizabilities):
                        temp = input_file.readline().split()
                        site_num = int(temp[0])
                        polarizability = [float(value) for value in temp[1:]]
                        setattr(potential[site_num], 'P{0[0]}{0[1]}'.format(polarizability_order), polarizability)
                else:
                    raise ValueError
            line = input_file.readline()
    return potential


def read_input_file(input_file, input_reader):
    return input_reader(input_file, FragmentDict, Fragment, AtomList, Atom)


class InputReaders(object):

    @staticmethod
    def pdb(filename, fragment_dict, fragment_class, atom_list, atom_class):
        """Read PDB input file and return fragment objects in fragments dictionary"""
        with open(filename, 'r') as input_file:
            fragments = fragment_dict()
            atom_number = 1
            line = input_file.readline()
            while line:
                if str(line[0:6]).strip() not in ['ATOM', 'HETATM']:
                    line = input_file.readline()
                    continue
                atom_type = str(line[0:6]).strip()
                try:
                    number = int(line[22:26])
                except:
                    raise PDBError('number', line)
                try:
                    chain_id = str(line[21]).strip()
                except:
                    raise PDBError('chain_id', line)
                try:
                    name = str(line[17:21]).strip()
                except:
                    raise PDBError('name', line)
                fragment = fragment_class()
                fragment.name = name
                fragment.number = number
                fragment.chain_id = chain_id
                if chain_id:
                    fragment.identifier = '{0}_{1}_{2}'.format(str(number), chain_id, name)
                else:
                    fragment.identifier = '{0}_{1}'.format(str(number), name)
                atoms = atom_list()
                while (atom_type == str(line[0:6]).strip() and number == int(line[22:26]) and
                       chain_id == str(line[21]).strip() and name == str(line[17:21]).strip()):
                    atom = atom_class()
                    try:
                        atom.name = str(line[12:16]).strip()
                    except:
                        raise PDBError('atom name', line)
                    try:
                        atom.coordinate = np.array([float(line[30:38]), float(line[38:46]),
                                                    float(line[46:54])])
                    except:
                        raise PDBError('coordinate', line)
                    if str(line[76:78]).strip():
                        atom.element = str(line[76:78]).strip()
                    else:
                        warnings.warn('No element was present in the pdb file. Guessing element based on atom name. Check the output carefully.')
                        if atom.name[0:2].title() in elements:
                            if fragment.name in amino_acid_names:
                                atom.element = atom.name[0].title()
                            else:
                                atom.element = atom.name[0:2].title()
                        elif atom.name[0].title() in elements:
                            atom.element = atom.name[0].title()
                        else:
                            raise PDBError('element', line)
                    if not str(line[78:80]).strip():
                        atom.charge = 0.0
                    elif str(line[79]).strip() in ['-', '+']:
                        try:
                            atom.charge = float(line[79] + line[78])
                        except:
                            raise PDBError('charge', line)
                    else:
                        try:
                            atom.charge = float(str(line[78:80]).strip())
                        except:
                            raise PDBError('charge', line)
                    atom.number = atom_number
                    atom_number += 1
                    atoms.append(atom)
                    line = input_file.readline()
                    if str(line[0:6]).strip() not in ['ATOM', 'HETATM']:
                        break
                    try:
                        str(line[0:6]).strip()
                    except:
                        raise PDBError('atom type', line)
                    try:
                        int(line[22:26])
                    except:
                        raise PDBError('number', line)
                    try:
                        str(line[21]).strip()
                    except:
                        raise PDBError('chain_id', line)
                    try:
                        str(line[17:21]).strip()
                    except:
                        raise PDBError('name', line)
                fragment.atoms = atoms
                fragments[fragment.identifier] = fragment
        return fragments

    @staticmethod
    def pqr(filename, fragment_dict, fragment_class, atom_list, atom_class):
        """Read PQR input file and return fragment objects in fragments dictionary"""
        with open(filename, 'r') as input_file:
            fragments = fragment_dict()
            atom_number = 1
            line = input_file.readline()
            while line:
                if str(line[0:6]).strip() not in ['ATOM', 'HETATM']:
                    line = input_file.readline()
                    continue
                atom_type = str(line[0:6]).strip()
                try:
                    number = int(line[22:26])
                except:
                    raise PDBError('number', line)
                try:
                    chain_id = str(line[21]).strip()
                except:
                    raise PDBError('chain_id', line)
                try:
                    name = str(line[17:21]).strip()
                except:
                    raise PDBError('name', line)
                fragment = fragment_class()
                fragment.name = name
                fragment.number = number
                fragment.chain_id = chain_id
                if chain_id:
                    fragment.identifier = '{0}_{1}_{2}'.format(str(number), chain_id, name)
                else:
                    fragment.identifier = '{0}_{1}'.format(str(number), name)
                atoms = atom_list()
                while (atom_type == str(line[0:6]).strip() and number == int(line[22:26]) and
                       chain_id == str(line[21]).strip() and name == str(line[17:21]).strip()):
                    atom = atom_class()
                    try:
                        atom.name = str(line[12:16]).strip()
                    except:
                        raise PDBError('atom name', line)
                    try:
                        atom.coordinate = np.array([float(line[30:38]), float(line[38:46]),
                                                    float(line[46:54])])
                    except:
                        raise PDBError('coordinate', line)
                    if str(line[76:78]).strip():
                        atom.element = str(line[76:78]).strip()
                    else:
                        warnings.warn('No element was present in the pdb file. Guessing element based on atom name. Check the output carefully.')
                        if atom.name[0:2].title() in elements:
                            if fragment.name in amino_acid_names:
                                atom.element = atom.name[0].title()
                            else:
                                atom.element = atom.name[0:2].title()
                        elif atom.name[0].title() in elements:
                            atom.element = atom.name[0].title()
                        else:
                            raise PDBError('element', line)
                    try:
                        atom.charge = float(line[54:60])
                    except:
                        raise PDBError('charge', line)
                    atom.number = atom_number
                    atom_number += 1
                    atoms.append(atom)
                    line = input_file.readline()
                    if str(line[0:6]).strip() not in ['ATOM', 'HETATM']:
                        break
                    try:
                        str(line[0:6]).strip()
                    except:
                        raise PDBError('atom type', line)
                    try:
                        int(line[22:26])
                    except:
                        raise PDBError('number', line)
                    try:
                        str(line[21]).strip()
                    except:
                        raise PDBError('chain_id', line)
                    try:
                        str(line[17:21]).strip()
                    except:
                        raise PDBError('name', line)
                fragment.atoms = atoms
                fragments[fragment.identifier] = fragment
        return fragments


class OutputReaders(object):

    @staticmethod
    def dalton_multipoles_polarizability(filename):
        """Reads output from Dalton LoProp calculations"""
        potential = {}
        with open('{0}.out'.format(filename)) as loprop:
            index = 1
            for line in loprop:
                if 'Nuclear charge' in line:
                    split_line = line.split()
                    nuclear_charge = int(round(float(split_line[2])))
                    element = elements[nuclear_charge - 1]
                    potential[index] = {}
                    potential[index]['element'] = element
                    index += 1
                if 'Molecular' in line:
                    break
        # the polarizability starts after an index, three coordinates and the multipole (1+3+x)
        with open('{0}.out'.format(filename)) as loprop:
            line = loprop.readline()
            unit = line
            line = loprop.readline().split()
            try:
                num_sites = int(line[0])
            except IndexError:
                raise IndexError(f'Failed to read {filename}.')
            except ValueError:
                raise ValueError(f'{filename} is empty?')
            multipole_order = int(line[1])
            # first we have the multipoles, then the polarizability
            # if we use -l -1 (multipole_order + 1) = 0, so offset will be correct
            # if we use -l -2 (multipole_order + 2) = 0, so offset will be correct
            # if we use -l -3 (multipole_order + 3) = 0, so offset will be correct
            pol_start = 1 + 3 + (multipole_order+1)*(multipole_order+2)*(multipole_order+3)//6
            pol_type = int(line[2])
            if num_sites != len(potential):
                raise ValueError(f'Inconsistency in {filename}.out')
            for i in range(1, num_sites + 1):
                line = loprop.readline().split()
                coordinate = [float(component) for component in line[1:4]]
                if 'AA' in unit:
                    pass
                elif 'AU' in unit:
                    coordinate = [float(component) * BOHR2AA for component in coordinate]
                else:
                    raise ValueError(f'unidentifiable unit in {filename}.out')
                potential[i]['coordinate'] = np.array(coordinate)
                if multipole_order >= 0:
                    M0 = [float(line[4])]
                    potential[i]['M0'] = M0
                if multipole_order >= 1:
                    M1 = [float(component) for component in line[5:8]]
                    potential[i]['M1'] = M1
                if multipole_order >= 2:
                    M2 = [float(component) for component in line[8:14]]
                    potential[i]['M2'] = M2
                if 2 >= pol_type > 0:
                    if pol_type == 1:
                        P11 = [float(line[pol_start]) * AA2BOHR**3]
                    elif pol_type == 2:
                        P11 = [float(component) * AA2BOHR**3 for component in line[pol_start:pol_start+6]]
                    potential[i]['P11'] = P11
        return potential

    @staticmethod
    def dalton_polarizability(*args):
        return OutputReaders.dalton_multipoles_polarizability(*args)

    @staticmethod
    def dalton_multipoles(*args):
        return OutputReaders.dalton_multipoles_polarizability(*args)

    @staticmethod
    def dalton_density_repulsion(filename):
        potential = {}
        index = 1
        h5file = pathlib.Path(filename).stem + '.h5'
        with h5py.File(h5file, 'r') as f:
            for i in range(len(f['fragment/charges'])):
                potential[index] = {}
                nuclear_charge = int(f['fragment/charges'][i])
                element = elements[nuclear_charge - 1]
                potential[index]['element'] = element
                potential[index]['coordinate'] = np.array(f['fragment/coordinates'][i])
                index += 1
        return potential

    @staticmethod
    def dalton_density(*args):
        return OutputReaders.dalton_density_repulsion(*args)

    @staticmethod
    def dalton_repulsion(*args):
        return OutputReaders.dalton_density_repulsion(*args)

    @staticmethod
    def dalton_multipoles_polarizability_repulsion(*args):
        return OutputReaders.dalton_multipoles_polarizability(*args)

    @staticmethod
    def dalton_polarizability_density_repulsion(*args):
        return OutputReaders.dalton_multipoles_polarizability(*args)


    @staticmethod
    def molcas_multipoles_polarizability(filename):
        """Reads output from MOLCAS LoProp calculations (MpProp file)"""
        potential = {}
        index = 1
        with open('{0}.out'.format(filename), 'r') as loprop:
            line = loprop.readline()
            while line:
                if '* Level of Multipoles and Polarizabilities' in line:
                    line = [int(order) for order in loprop.readline().split()]
                    multipole_order = line[0]
                if '    2    1    ' in line:
                    potential[index] = {}
                    line = line.split()
                    if '-' in line[2]:
                        element = 'X'
                    else:
                        element = ''.join(char for char in line[2] if not char.isdigit())
                    assert 1 <= len(element) <= 2
                    potential[index]['element'] = element
                    line = loprop.readline()
                    coordinate = [float(component) * BOHR2AA for component in line.split()]
                    potential[index]['coordinate'] = np.array(coordinate)
                    if multipole_order >= 0:
                        M0 = [float(loprop.readline())]
                        potential[index]['M0'] = M0
                    if multipole_order >= 1:
                        M1 = [float(component) for component in loprop.readline().split()]
                        potential[index]['M1'] = M1
                    if multipole_order >= 2:
                        M2 = [float(component) for component in loprop.readline().split()]
                        M2.extend([float(component) for component in loprop.readline().split()])
                        potential[index]['M2'] = M2
                    if multipole_order >= 3:
                        M3 = [float(component) for component in loprop.readline().split()]
                        for i in range(3):
                            M3.extend([float(component) for component in loprop.readline().split()])
                        potential[index]['M3'] = M3
                    if multipole_order >= 4:
                        M4 = [float(component) for component in loprop.readline().split()]
                        for i in range(4):
                            M4.extend([float(component) for component in loprop.readline().split()])
                        potential[index]['M4'] = M4
                    if multipole_order >= 5:
                        M5 = [float(component) for component in loprop.readline().split()]
                        for i in range(6):
                            M5.extend([float(component) for component in loprop.readline().split()])
                        potential[index]['M5'] = M5
                    if multipole_order >= 6:
                        M6 = [float(component) for component in loprop.readline().split()]
                        for i in range(9):
                            M6.extend([float(component) for component in loprop.readline().split()])
                        potential[index]['M6'] = M6
                    if multipole_order == 0:
                        loprop.readline()
                    elif multipole_order == 1:
                        for i in range(2):
                            loprop.readline()
                    elif multipole_order >= 2:
                        for i in range(4):
                            loprop.readline()
                    P11 = [float(component) for component in loprop.readline().split()]
                    P11.extend([float(component) for component in loprop.readline().split()])
                    potential[index]['P11'] = P11
                    index += 1
                line = loprop.readline()
        return potential

    @staticmethod
    def molcas_polarizability(*args):
        return OutputReaders.molcas_multipoles_polarizability(*args)

    @staticmethod
    def molcas_multipoles(*args):
        return OutputReaders.molcas_multipoles_polarizability(*args)


class PDBError(Exception):
    """Exception for errors in PDB file."""

    def __init__(self, prop, line):
        self.prop = prop
        self.line = line

    def __str__(self):
        return 'unable to read {0} from:\n{1}'.format(self.prop, self.line)
