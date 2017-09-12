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
        self.name = ''
        self.number = None
        self.element = ''
        self.charge = None
        self.coordinate = None
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
    def mass(self):
        if self._mass is None:
            self._mass = element2mass[self.element]
        return self._mass

    @mass.setter
    def mass(self, mass):
        assert isinstance(mass, float)
        self._mass = mass

    def copy(self):
        return Atom(name=self.name, coordinate=self.coordinate, element=self.element,
                    charge=self.charge, number=self.number)