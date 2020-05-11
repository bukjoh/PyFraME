"""Tests for atoms module"""

import pytest
import numpy as np
import io

import pyframe.atoms as atoms


def test_atom_init():
    atom = atoms.Atom(name='one', number=1, element='C', charge=-1.0, coordinate=[1.1, 2.3, -1.5])
    assert isinstance(atom, atoms.Atom)
    assert isinstance(atom.name, str)
    assert atom.name == 'one'
    assert isinstance(atom.number, int)
    assert atom.number == 1
    assert isinstance(atom.element, str)
    assert atom.element == 'C'
    assert isinstance(atom.charge, float)
    assert atom.charge == -1.0
    assert isinstance(atom.coordinate, np.ndarray)
    assert atom.coordinate.tolist() == [1.1, 2.3, -1.5]
    assert isinstance(atom.mass, float)
    assert atom.mass == 12.0
    assert isinstance(atom.radius, float)
    assert atom.radius == 0.76
    with pytest.raises(ValueError):
        atoms.Atom(test='test')


def test_atom_setattr():
    atom = atoms.Atom()
    assert atom.name is None
    assert atom.number is None
    assert atom.element is None
    assert atom.charge is None
    assert atom.coordinate is None
    assert atom.mass is None
    assert atom.radius is None
    atom.name = 'one'
    assert atom.name == 'one'
    with pytest.raises(TypeError):
        atom.name = ['one']
    atom.number = 1
    assert atom.number == 1
    with pytest.raises(TypeError):
        atom.number = 1.0
    with pytest.raises(ValueError):
        atom.number = -1
    atom.element = 'C'
    assert atom.element == 'C'
    assert atom.mass == 12.0
    assert atom.radius == 0.76
    with pytest.raises(TypeError):
        atom.element = 6
    with pytest.raises(ValueError):
        atom.element = 'X'
    atom.charge = -1.0
    assert atom.charge == -1.0
    with pytest.raises(TypeError):
        atom.charge = -1
    atom.coordinate = [1.1, 2.3, -1.5]
    assert isinstance(atom.coordinate, np.ndarray)
    assert atom.coordinate.tolist() == [1.1, 2.3, -1.5]
    atom.coordinate = np.array([1.1, 2.3, -1.5])
    assert isinstance(atom.coordinate, np.ndarray)
    assert atom.coordinate.tolist() == [1.1, 2.3, -1.5]
    with pytest.raises(TypeError):
        atom.coordinate = (0.0, 0.0, -1.0)
    atom.mass = 13.0
    assert atom.mass == 13.0
    with pytest.raises(TypeError):
        atom.mass = 13
    atom.radius = 0.5
    assert atom.radius == 0.5
    with pytest.raises(TypeError):
        atom.radius = 1


def test_atom_str():
    atom = atoms.Atom(name='one', number=1, element='C', charge=-1.0, coordinate=[1.1, 2.3, -1.5])
    test_output = io.StringIO()
    print(atom, file=test_output)
    ref_output = 'atom name=one, number=1, element=C, charge=-1.0, coordinate=[ 1.1  2.3 -1.5] mass=12.0 radius=0.76\n'
    assert test_output.getvalue() == ref_output


def test_atom_copy():
    atom = atoms.Atom(name='one', number=1, element='C', charge=-1.0, coordinate=[1.1, 2.3, -1.5])
    atom_copy = atom.copy()
    assert atom.name == atom_copy.name
    assert atom.number == atom_copy.number
    assert atom.element == atom_copy.element
    assert atom.charge == atom_copy.charge
    assert atom.coordinate.all() == atom_copy.coordinate.all()
    assert atom.mass == atom_copy.mass
    assert atom.radius == atom_copy.radius


def test_atomlist_init():
    atom_one = atoms.Atom(name='one', number=1, element='C', charge=-1.0, coordinate=[1.1, 2.3, -1.5])
    atom_two = atoms.Atom(name='two', number=2, element='O', charge=1.0, coordinate=[1.1, 2.3, 0.0])
    atom_three = atoms.Atom(name='three', number=3, element='O', charge=-1.0, coordinate=[1.1, 2.3, 1.5])
    first_atomlist = atoms.AtomList([atom_one, atom_two, atom_three])
    second_atomlist = atoms.AtomList([atom_one, atom_two, atom_three])
    assert first_atomlist == second_atomlist
    # atoms = atoms.AtomList((atom_one))


def test_atomlist_contains():
    atom_one = atoms.Atom(name='one', number=1, element='C', charge=-1.0, coordinate=[1.1, 2.3, -1.5])
    atom_two = atoms.Atom(name='two', number=2, element='O', charge=1.0, coordinate=[1.1, 2.3, 0.0])
    atom_three = atoms.Atom(name='three', number=3, element='O', charge=-1.0, coordinate=[1.1, 2.3, 1.5])
    atomlist = atoms.AtomList([atom_one, atom_two, atom_three])
    assert 2 in atomlist
    assert 20 not in atomlist


def test_atomlist_copy():
    atom_one = atoms.Atom(name='one', number=1, element='C', charge=-1.0, coordinate=[1.1, 2.3, -1.5])
    atom_two = atoms.Atom(name='two', number=2, element='O', charge=1.0, coordinate=[1.1, 2.3, 0.0])
    atom_three = atoms.Atom(name='three', number=3, element='O', charge=-1.0, coordinate=[1.1, 2.3, 1.5])
    atomlist = atoms.AtomList([atom_one, atom_two, atom_three])
    atomlist_copy = atomlist.copy()
    assert atomlist is not atomlist_copy
    assert len(atomlist) == len(atomlist_copy)
    for atom, atom_copy in zip(atomlist, atomlist_copy):
        assert atom.name == atom_copy.name
        assert atom.number == atom_copy.number
        assert atom.element == atom_copy.element
        assert atom.charge == atom_copy.charge
        assert atom.coordinate.all() == atom_copy.coordinate.all()
        assert atom.mass == atom_copy.mass
        assert atom.radius == atom_copy.radius


def test_atomlist_index():
    atom_one = atoms.Atom(name='one', number=1, element='C', charge=-1.0, coordinate=[1.1, 2.3, -1.5])
    atom_two = atoms.Atom(name='two', number=2, element='O', charge=1.0, coordinate=[1.1, 2.3, 0.0])
    atom_three = atoms.Atom(name='three', number=3, element='O', charge=-1.0, coordinate=[1.1, 2.3, 1.5])
    atom_four = atoms.Atom(name='four', number=4, element='C', charge=1.0, coordinate=[1.1, 2.3, 2.5])
    atomlist = atoms.AtomList([atom_one, atom_two, atom_three, atom_four])
    assert atomlist.index(2) == 1
    assert atomlist.index(3, start=2, stop=4) == 2
    with pytest.raises(ValueError):
        atomlist.index(5)


def test_atomlist_pop():
    atom_one = atoms.Atom(name='one', number=1, element='C', charge=-1.0, coordinate=[1.1, 2.3, -1.5])
    atom_two = atoms.Atom(name='two', number=2, element='O', charge=1.0, coordinate=[1.1, 2.3, 0.0])
    atom_three = atoms.Atom(name='three', number=3, element='O', charge=-1.0, coordinate=[1.1, 2.3, 1.5])
    atom_four = atoms.Atom(name='four', number=4, element='C', charge=1.0, coordinate=[1.1, 2.3, 2.5])
    atomlist = atoms.AtomList([atom_one, atom_two, atom_three, atom_four])
    assert len(atomlist) == 4
    atom = atomlist.pop()
    with pytest.raises(ValueError):
        atomlist.index(4)
    assert len(atomlist) == 3
    assert atom.name == atom_four.name
    assert atom.number == atom_four.number
    assert atom.element == atom_four.element
    assert atom.charge == atom_four.charge
    assert atom.coordinate.all() == atom_four.coordinate.all()
    assert atom.mass == atom_four.mass
    assert atom.radius == atom_four.radius
    atom = atomlist.pop(2)
    assert len(atomlist) == 2
    with pytest.raises(ValueError):
        atomlist.index(2)
    assert atom.name == atom_two.name
    assert atom.number == atom_two.number
    assert atom.element == atom_two.element
    assert atom.charge == atom_two.charge
    assert atom.coordinate.all() == atom_two.coordinate.all()
    assert atom.mass == atom_two.mass
    assert atom.radius == atom_two.radius


def test_atomlist_get():
    atom_one = atoms.Atom(name='one', number=1, element='C', charge=-1.0, coordinate=[1.1, 2.3, -1.5])
    atom_two = atoms.Atom(name='two', number=2, element='O', charge=1.0, coordinate=[1.1, 2.3, 0.0])
    atom_three = atoms.Atom(name='three', number=3, element='O', charge=-1.0, coordinate=[1.1, 2.3, 1.5])
    atom_four = atoms.Atom(name='four', number=4, element='C', charge=1.0, coordinate=[1.1, 2.3, 2.5])
    atomlist = atoms.AtomList([atom_one, atom_two, atom_three, atom_four])
    atom = atomlist.get(2)
    assert atom.name == atom_two.name
    assert atom.number == atom_two.number
    assert atom.element == atom_two.element
    assert atom.charge == atom_two.charge
    assert atom.coordinate.all() == atom_two.coordinate.all()
    assert atom.mass == atom_two.mass
    assert atom.radius == atom_two.radius
