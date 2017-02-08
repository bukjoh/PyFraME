# coding=utf-8
"""Blablabla"""

import numpy as np

from .utils import element2mass

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

    def __init__(self, **kwargs):
        self._name = ''
        self._number = None
        self._element = ''
        self._charge = None
        self._coordinate = None
        self._mass = None
        for key in kwargs.keys():
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            else:
                # TODO: replace exit with exception
                exit('ERROR: unknown atom property "{0}"'.format(key))

    def __str__(self):
        msg = 'atom name={0},'.format(self.name)
        msg += ' number={0},'.format(self.number)
        msg += ' element={0},'.format(self.element)
        msg += ' charge={0},'.format(self.charge)
        msg += ' coordinate={0}'.format(self.coordinate)
        return msg

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        assert isinstance(name, str)
        self._name = name

    @property
    def coordinate(self):
        return self._coordinate

    @coordinate.setter
    def coordinate(self, coordinate):
        assert all(isinstance(coord, float) for coord in coordinate)
        assert len(coordinate) == 3
        self._coordinate = np.array(coordinate)

    @property
    def element(self):
        return self._element

    @element.setter
    def element(self, element):
        assert isinstance(element, str)
        self._element = element
        self._mass = element2mass[self.element]

    @property
    def charge(self):
        return self._charge

    @charge.setter
    def charge(self, charge):
        assert isinstance(charge, float)
        self._charge = charge

    @property
    def mass(self):
        return self._mass

    @property
    def number(self):
        return self._number

    @number.setter
    def number(self, number):
        assert isinstance(number, int)
        self._number = number

    def copy(self):
        return Atom(name=self.name, coordinate=self.coordinate, element=self.element,
                    charge=self.charge, number=self.number)