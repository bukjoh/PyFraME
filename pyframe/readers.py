# coding=utf-8
"""Blablabla"""

import os.path
import collections

import numpy as np

from .fragments import FragmentDict, Fragment
from .atoms import AtomList, Atom
from .potentials import PotentialDict, Potential
from .utils import BOHR2AA, AA2BOHR, elements
from .errors import PDBError


__all__ = ['OutputReaders', 'InputReaders', 'read_potential_file', 'read_pelib_potential', 'read_input_file']


def read_potential_file(potential_file):

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
        potential = collections.OrderedDict()
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
    def read_pdb(filename, fragment_dict, fragment_class, atom_list, atom_class):
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
                    try:
                        atom.element = str(line[76:78]).strip()
                    except:
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
    def read_pqr(filename, fragment_dict, fragment_class, atom_list, atom_class):
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
                    try:
                        atom.element = str(line[76:78]).strip()
                    except:
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
    def dalton_loprop(filename):
        """Reads output from Dalton LoProp calculations"""
        potential = collections.OrderedDict()
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
        with open('{0}.out'.format(filename)) as loprop:
            line = loprop.readline()
            unit = line
            line = loprop.readline().split()
            n_sites = int(line[0])
            multipole_order = int(line[1])
            pol_type = int(line[2])
            if n_sites != len(potential):
                # TODO: replace with exception
                exit('ERROR: inconsistency in {0}.out'.format(filename))
            for i in range(1, n_sites + 1):
                line = loprop.readline().split()
                coordinate = [float(component) for component in line[1:4]]
                if 'AA' in unit:
                    pass
                elif 'AU' in unit:
                    coordinate = [float(component) * BOHR2AA for component in coordinate]
                else:
                    # TODO: replace with exception
                    exit('ERROR: unidentifiable unit in {0}.out'.format(filename))
                potential[i]['coordinate'] = np.array(coordinate)
                if multipole_order >= 0:
                    M0 = [float(line[4])]
                    if not all(abs(component) < 1.0e-6 for component in M0):
                        potential[i]['M0'] = M0
                if multipole_order >= 1:
                    M1 = [float(component) for component in line[5:8]]
                    if not all(abs(component) < 1.0e-6 for component in M1):
                        potential[i]['M1'] = M1
                if multipole_order >= 2:
                    M2 = [float(component) for component in line[8:14]]
                    if not all(abs(component) < 1.0e-6 for component in M2):
                        potential[i]['M2'] = M2
                if 2 >= pol_type > 0:
                    if pol_type == 1:
                        P11 = [float(line[14]) * AA2BOHR**3]
                    elif pol_type == 2:
                        P11 = [float(component) * AA2BOHR**3 for component in line[14:20]]
                    if not all(abs(component) < 1.0e-6 for component in P11):
                        potential[i]['P11'] = P11
        return potential

    @staticmethod
    def dalton_loprop_polarizability(*args):
        return OutputReaders.molcas_loprop(*args)

    @staticmethod
    def dalton_loprop_multipoles(*args):
        return OutputReaders.dalton_loprop(*args)

    @staticmethod
    def molcas_loprop(filename):
        """Reads output from MOLCAS LoProp calculations (MpProp file)"""
        potential = collections.OrderedDict()
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
                        if not all(abs(component) < 1.0e-6 for component in M0):
                            potential[index]['M0'] = M0
                    if multipole_order >= 1:
                        M1 = [float(component) for component in loprop.readline().split()]
                        if not all(abs(component) < 1.0e-6 for component in M1):
                            potential[index]['M1'] = M1
                    if multipole_order >= 2:
                        M2 = [float(component) for component in loprop.readline().split()]
                        M2.extend([float(component) for component in loprop.readline().split()])
                        if not all(abs(component) < 1.0e-6 for component in M2):
                            potential[index]['M2'] = M2
                    if multipole_order >= 3:
                        M3 = [float(component) for component in loprop.readline().split()]
                        for i in range(3):
                            M3.extend([float(component) for component in loprop.readline().split()])
                        if not all(abs(component) < 1.0e-6 for component in M3):
                            potential[index]['M3'] = M3
                    if multipole_order >= 4:
                        M4 = [float(component) for component in loprop.readline().split()]
                        for i in range(4):
                            M4.extend([float(component) for component in loprop.readline().split()])
                        if not all(abs(component) < 1.0e-6 for component in M4):
                            potential[index]['M4'] = M4
                    if multipole_order >= 5:
                        M5 = [float(component) for component in loprop.readline().split()]
                        for i in range(6):
                            M5.extend([float(component) for component in loprop.readline().split()])
                        if not all(abs(component) < 1.0e-6 for component in M5):
                            potential[index]['M5'] = M5
                    if multipole_order >= 6:
                        M6 = [float(component) for component in loprop.readline().split()]
                        for i in range(9):
                            M6.extend([float(component) for component in loprop.readline().split()])
                        if not all(abs(component) < 1.0e-6 for component in M6):
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
                    if not all(abs(component) < 1.0e-6 for component in P11):
                        potential[index]['P11'] = P11
                    index += 1
                line = loprop.readline()
        return potential

    @staticmethod
    def molcas_loprop_polarizability(*args):
        return OutputReaders.molcas_loprop(*args)

    @staticmethod
    def molcas_loprop_multipoles(*args):
        return OutputReaders.molcas_loprop(*args)
