# coding=utf-8
"""Tests PyFraME.utils.py"""

import unittest

import numpy as np

import pyframe.atoms as atoms
import pyframe.utils as utils


class TestUtils(unittest.TestCase):

    def test_element2radius(self):
        for element in utils.elements:
            self.assertIsInstance(element, str)
            self.assertIsInstance(utils.element2radius[element], float)

    def test_element2charge(self):
        for element in utils.elements:
            self.assertIsInstance(element, str)
            self.assertIsInstance(utils.element2charge[element], int)

    def test_element2mass(self):
        for element in utils.elements:
            self.assertIsInstance(element, str)
            self.assertIsInstance(utils.element2mass[element], float)

    def test_get_angle(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        c = np.array([0.0, 0.0, 0.0])
        angle_rad = utils.compute_angle(a, b, c)
        angle_deg = np.rad2deg(angle_rad)
        self.assertIsInstance(angle_rad, float)
        self.assertAlmostEqual(angle_deg, 45.0)

    def test_get_bond_length(self):
        for first in utils.elements:
            for second in utils.elements:
                r = utils.get_bond_length(first, second)
                self.assertIsInstance(r, float)
                # because these values are used in PyFraME/fragments.py
                self.assertLess(r, 5.4)
                self.assertGreater(r, 0.5)
                if first == 'H':
                    self.assertLess(r, 3.0)

    def test_scale_bond_length(self):
        first_atom = atoms.Atom(coordinate=np.array([1.0, 0.0, 0.0]), element='H')
        second_atom = atoms.Atom(coordinate=np.array([2.0, 1.0, 0.0]), element='C')
        coordinate = utils.scale_bond_length(first_atom, second_atom)
        print(coordinate)


if __name__ == '__main__':
    unittest.main()
