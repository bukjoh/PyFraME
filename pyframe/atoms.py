# coding=utf-8
"""Blablabla"""

import numpy as np

from .utils import elements, element2radius, element2mass

__all__ = ['AtomList', 'Atom']


class AtomList(list):

    """Atom dictionary methods"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __contains__(self, number):
        for atom in self:
            if atom.number == number:
                return True
        return False

    def copy(self):
        return AtomList(atom.copy() for atom in self)

    def index(self, number, start=None, stop=None):
        if start is None:
            start = 0
        if stop is None:
            stop = len(self)
        for index, atom in enumerate(self[start:stop]):
            if atom.number == number:
                return index + start
        raise ValueError('atom number {0} not found in atom list'.format(number))

    def pop(self, number=None):
        if number:
            index = self.index(number)
        else:
            index = -1
        atom = self[index]
        del self[index]
        return atom

    def get(self, number):
        index = self.index(number)
        return self[index]


class Atom(object):

    """Container for atom attributes and methods"""

    name = None
    number = None
    element = None
    charge = None
    coordinate = None
    mass = None
    radius = None

    def __init__(self, **kwargs):
        for key in kwargs.keys():
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            else:
                raise ValueError('ERROR: unknown atom property "{0}"'.format(key))

    def __str__(self):
        msg = 'atom name={0},'.format(self.name)
        msg += ' number={0},'.format(self.number)
        msg += ' element={0},'.format(self.element)
        msg += ' charge={0},'.format(self.charge)
        msg += ' coordinate={0}'.format(self.coordinate)
        msg += ' mass={0}'.format(self.mass)
        msg += ' radius={0}'.format(self.radius)
        return msg

    def __setattr__(self, attribute, value):
        if attribute == 'name':
            if not isinstance(value, str):
                raise TypeError
        elif attribute == 'number':
            if not isinstance(value, int):
                raise TypeError
            elif value < 0:
                raise ValueError
        elif attribute == 'element':
            if not isinstance(value, str):
                raise TypeError
            elif value not in elements:
                raise ValueError
            self.mass = element2mass[value]
            self.radius = element2radius[value]
        elif attribute == 'charge':
            if not isinstance(value, float):
                raise TypeError
        elif attribute == 'coordinate':
            if isinstance(value, list):
                value = np.array(value)
            elif isinstance(value, np.ndarray):
                pass
            else:
                raise TypeError
        elif attribute == 'mass':
            if not isinstance(value, float):
                raise TypeError
        elif attribute == 'radius':
            if not isinstance(value, float):
                raise TypeError
        super(Atom, self).__setattr__(attribute, value)

    def copy(self):
        return Atom(name=self.name, number=self.number, element=self.element, charge=self.charge,
                    coordinate=self.coordinate)
