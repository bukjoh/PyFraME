"""Tests PyFraME.utils.py"""

import pytest
import numpy as np

import pyframe.atoms as atoms
import pyframe.utils as utils


def test_element2radius():
    for element in utils.elements:
        assert isinstance(element, str)
        assert isinstance(utils.element2radius[element], float)


def test_element2charge():
    for element in utils.elements:
        assert isinstance(element, str)
        assert isinstance(utils.element2charge[element], int)


def test_element2mass():
    for element in utils.elements:
        assert isinstance(element, str)
        assert isinstance(utils.element2mass[element], float)


def test_compute_distance():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    distance = utils.compute_distance(a, b)
    assert pytest.approx(distance) == np.sqrt(2.0)


def test_compute_angle():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([0.0, 0.0, 0.0])
    angle_rad = utils.compute_angle(a, b, c)
    angle_deg = np.rad2deg(angle_rad)
    assert isinstance(angle_rad, float)
    assert pytest.approx(angle_deg) == 45.0


def test_get_bond_length():
    for first in utils.elements:
        for second in utils.elements:
            r = utils.get_bond_length(first, second)
            assert isinstance(r, float)
            # because these values are used in PyFraME/fragments.py
            assert r < 5.4
            assert r > 0.5
            if first == 'H':
                assert r < 3.0


def test_scale_bond_length():
    first_atom = atoms.Atom(coordinate=np.array([1.0, 0.0, 0.0]), element='H')
    second_atom = atoms.Atom(coordinate=np.array([2.0, 1.0, 0.0]), element='C')
    coordinate = utils.scale_bond_length(first_atom, second_atom)
    assert pytest.approx(coordinate) == [1.75660426, 0.75660426, 0.0]
