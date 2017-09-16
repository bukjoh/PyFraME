# coding=utf-8
"""Tests for atoms module"""

import unittest
import numpy as np
import io

import pyframe.atoms as atoms


class TestAtoms(unittest.TestCase):

    def test_atom_init(self):
        atom = atoms.Atom(name='one', number=1, element='C', charge=-1.0, coordinate=[1.1, 2.3, -1.5])
        self.assertIs(type(atom), atoms.Atom)
        self.assertIs(type(atom.name), str)
        self.assertEqual(atom.name, 'one')
        self.assertIs(type(atom.number), int)
        self.assertEqual(atom.number, 1)
        self.assertIs(type(atom.element), str)
        self.assertEqual(atom.element, 'C')
        self.assertIs(type(atom.charge), float)
        self.assertEqual(atom.charge, -1.0)
        self.assertIs(type(atom.coordinate), np.ndarray)
        self.assertListEqual(atom.coordinate.tolist(), [1.1, 2.3, -1.5])
        self.assertIs(type(atom.mass), float)
        self.assertEqual(atom.mass, 12.0)
        self.assertIs(type(atom.radius), float)
        self.assertEqual(atom.radius, 0.76)
        with self.assertRaises(ValueError):
            atoms.Atom(test='test')

    def test_atom_setattr(self):
        atom = atoms.Atom()
        self.assertIsNone(atom.name)
        self.assertIsNone(atom.number)
        self.assertIsNone(atom.element)
        self.assertIsNone(atom.charge)
        self.assertIsNone(atom.coordinate)
        self.assertIsNone(atom.mass)
        self.assertIsNone(atom.radius)
        atom.name = 'one'
        self.assertEqual(atom.name, 'one')
        with self.assertRaises(TypeError):
            atom.name = ['one']
        atom.number = 1
        self.assertEqual(atom.number, 1)
        with self.assertRaises(TypeError):
            atom.number = 1.0
        with self.assertRaises(ValueError):
            atom.number = -1
        atom.element = 'C'
        self.assertEqual(atom.element, 'C')
        self.assertEqual(atom.mass, 12.0)
        self.assertEqual(atom.radius, 0.76)
        with self.assertRaises(TypeError):
            atom.element = 6
        with self.assertRaises(ValueError):
            atom.element = 'X'
        atom.charge = -1.0
        self.assertEqual(atom.charge, -1.0)
        with self.assertRaises(TypeError):
            atom.charge = -1
        atom.coordinate = [1.1, 2.3, -1.5]
        self.assertIs(type(atom.coordinate), np.ndarray)
        self.assertListEqual(atom.coordinate.tolist(), [1.1, 2.3, -1.5])
        atom.coordinate = np.array([1.1, 2.3, -1.5])
        self.assertIs(type(atom.coordinate), np.ndarray)
        self.assertListEqual(atom.coordinate.tolist(), [1.1, 2.3, -1.5])
        with self.assertRaises(TypeError):
            atom.coordinate = (0.0, 0.0, -1.0)
        atom.mass = 13.0
        self.assertEqual(atom.mass, 13.0)
        with self.assertRaises(TypeError):
            atom.mass = 13
        atom.radius = 0.5
        self.assertEqual(atom.radius, 0.5)
        with self.assertRaises(TypeError):
            atom.radius = 1

    def test_atom_str(self):
        atom = atoms.Atom(name='one', number=1, element='C', charge=-1.0, coordinate=[1.1, 2.3, -1.5])
        test_output = io.StringIO()
        print(atom, file=test_output)
        ref_output = 'atom name=one, number=1, element=C, charge=-1.0, coordinate=[ 1.1  2.3 -1.5] mass=12.0 radius=0.76\n'
        self.assertEqual(test_output.getvalue(), ref_output)

    def test_atom_copy(self):
        atom = atoms.Atom(name='one', number=1, element='C', charge=-1.0, coordinate=[1.1, 2.3, -1.5])
        atom_copy = atom.copy()
        self.assertEqual(atom.name, atom_copy.name)
        self.assertEqual(atom.number, atom_copy.number)
        self.assertEqual(atom.element, atom_copy.element)
        self.assertEqual(atom.charge, atom_copy.charge)
        self.assertListEqual(atom.coordinate.tolist(), atom_copy.coordinate.tolist())
        self.assertEqual(atom.mass, atom_copy.mass)
        self.assertEqual(atom.radius, atom_copy.radius)

    def test_atomlist_init(self):
        atom_one = atoms.Atom(name='one', number=1, element='C', charge=-1.0, coordinate=[1.1, 2.3, -1.5])
        atom_two = atoms.Atom(name='two', number=2, element='O', charge=1.0, coordinate=[1.1, 2.3, 0.0])
        atom_three = atoms.Atom(name='three', number=3, element='O', charge=-1.0, coordinate=[1.1, 2.3, 1.5])
        first_atomlist = atoms.AtomList([atom_one, atom_two, atom_three])
        second_atomlist = atoms.AtomList([atom_one, atom_two, atom_three])
        self.assertListEqual(first_atomlist, second_atomlist)
        #atoms = atoms.AtomList((atom_one))

    def test_atomlist_contains(self):
        atom_one = atoms.Atom(name='one', number=1, element='C', charge=-1.0, coordinate=[1.1, 2.3, -1.5])
        atom_two = atoms.Atom(name='two', number=2, element='O', charge=1.0, coordinate=[1.1, 2.3, 0.0])
        atom_three = atoms.Atom(name='three', number=3, element='O', charge=-1.0, coordinate=[1.1, 2.3, 1.5])
        atomlist = atoms.AtomList([atom_one, atom_two, atom_three])
        self.assertIn(2, atomlist)
        self.assertNotIn(20, atomlist)

    def test_atomlist_copy(self):
        atom_one = atoms.Atom(name='one', number=1, element='C', charge=-1.0, coordinate=[1.1, 2.3, -1.5])
        atom_two = atoms.Atom(name='two', number=2, element='O', charge=1.0, coordinate=[1.1, 2.3, 0.0])
        atom_three = atoms.Atom(name='three', number=3, element='O', charge=-1.0, coordinate=[1.1, 2.3, 1.5])
        atomlist = atoms.AtomList([atom_one, atom_two, atom_three])
        atomlist_copy = atomlist.copy()
        self.assertIsNot(atomlist, atomlist_copy)
        self.assertEqual(len(atomlist), len(atomlist_copy))
        for atom, atom_copy in zip(atomlist, atomlist_copy):
            self.assertEqual(atom.name, atom_copy.name)
            self.assertEqual(atom.number, atom_copy.number)
            self.assertEqual(atom.element, atom_copy.element)
            self.assertEqual(atom.charge, atom_copy.charge)
            self.assertEqual(atom.coordinate.all(), atom_copy.coordinate.all())
            self.assertEqual(atom.mass, atom_copy.mass)
            self.assertEqual(atom.radius, atom_copy.radius)

    def test_atomlist_index(self):
        atom_one = atoms.Atom(name='one', number=1, element='C', charge=-1.0, coordinate=[1.1, 2.3, -1.5])
        atom_two = atoms.Atom(name='two', number=2, element='O', charge=1.0, coordinate=[1.1, 2.3, 0.0])
        atom_three = atoms.Atom(name='three', number=3, element='O', charge=-1.0, coordinate=[1.1, 2.3, 1.5])
        atom_four = atoms.Atom(name='four', number=4, element='C', charge=1.0, coordinate=[1.1, 2.3, 2.5])
        atomlist = atoms.AtomList([atom_one, atom_two, atom_three, atom_four])
        self.assertEqual(atomlist.index(2), 1)
        self.assertEqual(atomlist.index(3, start=2, stop=4), 2)
        with self.assertRaises(ValueError):
            atomlist.index(5)

    def test_atomlist_pop(self):
        atom_one = atoms.Atom(name='one', number=1, element='C', charge=-1.0, coordinate=[1.1, 2.3, -1.5])
        atom_two = atoms.Atom(name='two', number=2, element='O', charge=1.0, coordinate=[1.1, 2.3, 0.0])
        atom_three = atoms.Atom(name='three', number=3, element='O', charge=-1.0, coordinate=[1.1, 2.3, 1.5])
        atom_four = atoms.Atom(name='four', number=4, element='C', charge=1.0, coordinate=[1.1, 2.3, 2.5])
        atomlist = atoms.AtomList([atom_one, atom_two, atom_three, atom_four])
        self.assertEqual(len(atomlist), 4)
        atom = atomlist.pop()
        with self.assertRaises(ValueError):
            atomlist.index(4)
        self.assertEqual(len(atomlist), 3)
        self.assertEqual(atom.name, atom_four.name)
        self.assertEqual(atom.number, atom_four.number)
        self.assertEqual(atom.element, atom_four.element)
        self.assertEqual(atom.charge, atom_four.charge)
        self.assertEqual(atom.coordinate.all(), atom_four.coordinate.all())
        self.assertEqual(atom.mass, atom_four.mass)
        self.assertEqual(atom.radius, atom_four.radius)
        atom = atomlist.pop(2)
        self.assertEqual(len(atomlist), 2)
        with self.assertRaises(ValueError):
            atomlist.index(2)
        self.assertEqual(atom.name, atom_two.name)
        self.assertEqual(atom.number, atom_two.number)
        self.assertEqual(atom.element, atom_two.element)
        self.assertEqual(atom.charge, atom_two.charge)
        self.assertEqual(atom.coordinate.all(), atom_two.coordinate.all())
        self.assertEqual(atom.mass, atom_two.mass)
        self.assertEqual(atom.radius, atom_two.radius)

    def test_atomlist_get(self):
        atom_one = atoms.Atom(name='one', number=1, element='C', charge=-1.0, coordinate=[1.1, 2.3, -1.5])
        atom_two = atoms.Atom(name='two', number=2, element='O', charge=1.0, coordinate=[1.1, 2.3, 0.0])
        atom_three = atoms.Atom(name='three', number=3, element='O', charge=-1.0, coordinate=[1.1, 2.3, 1.5])
        atom_four = atoms.Atom(name='four', number=4, element='C', charge=1.0, coordinate=[1.1, 2.3, 2.5])
        atomlist = atoms.AtomList([atom_one, atom_two, atom_three, atom_four])
        atom = atomlist.get(2)
        self.assertEqual(atom.name, atom_two.name)
        self.assertEqual(atom.number, atom_two.number)
        self.assertEqual(atom.element, atom_two.element)
        self.assertEqual(atom.charge, atom_two.charge)
        self.assertEqual(atom.coordinate.all(), atom_two.coordinate.all())
        self.assertEqual(atom.mass, atom_two.mass)
        self.assertEqual(atom.radius, atom_two.radius)
