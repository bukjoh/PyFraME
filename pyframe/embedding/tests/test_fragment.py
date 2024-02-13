"""Tests PyFraME.embedding.fragment.py"""
import numpy as np
import pytest
from pyframe.embedding import fragment, density_matrix, particle

def test_init_fragment():
    test_fragment = fragment.Fragment(index=1, name="H2O")
    assert test_fragment._index == 1
    assert test_fragment._name == "H2O"

class TestClassicalFragment:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Common setup for ClassicalFragment tests
        self.index = 0
        self.atoms = [{'element': 'H', 'coordinate': [0, 0, 0], 'index': 0, 'multipoles': {"elements":[0]}},
                 {'element': 'O', 'coordinate': [1, 1, 1], 'index': 1, 'multipoles': {"elements":[0]}}]
        self.test_fragment = fragment.ClassicalFragment(self.index, self.atoms)
    def test_initialization(self,
                            oxygen_atom_in_atoms_list
                            ):
        assert self.test_fragment.num_atoms == 2
        test_fragment2 = fragment.ClassicalFragment(index=1, atoms=oxygen_atom_in_atoms_list, name="H2O")
        assert test_fragment2._name == "H2O"
        assert len(test_fragment2.atoms) == 1
        assert isinstance(test_fragment2.atoms[0], particle.Atom)

    def test_potential_calculation(self,
                                   water_fragments
                                   ):
        coordinate = np.array([0, 0, 0])
        potential = self.test_fragment.potential(coordinate)
        assert isinstance(potential, float) or isinstance(potential, np.ndarray)
        water_fragment_1 = water_fragments[0]
        water_fragment_2 = water_fragments[1]
        # Potential
        ref_potential = 0
        for frag_atoms in water_fragment_1.atoms:
            ref_potential += frag_atoms.potential(coordinate=water_fragment_2.atoms[0].coordinate)
        assert water_fragment_1.potential(coordinate=water_fragment_2.atoms[0].coordinate) == ref_potential
        ref_potential = 0
        for frag_atoms in water_fragment_2.atoms:
            ref_potential += frag_atoms.potential(coordinate=water_fragment_1.atoms[0].coordinate)
        assert water_fragment_2.potential(coordinate=water_fragment_1.atoms[0].coordinate) == ref_potential
        # First derivative of the Potential
        ref_potential = 0
        for frag_atoms in water_fragment_1.atoms:
            ref_potential += frag_atoms.potential(coordinate=water_fragment_2.atoms[0].coordinate,
                                                  pot_derivative_order=1)
        assert np.allclose(water_fragment_1.potential(coordinate=water_fragment_2.atoms[0].coordinate,
                                                      pot_derivative_order=1), ref_potential)
        ref_potential = 0
        for frag_atoms in water_fragment_2.atoms:
            ref_potential += frag_atoms.potential(coordinate=water_fragment_1.atoms[0].coordinate,
                                                  pot_derivative_order=1)
        assert np.allclose(water_fragment_2.potential(coordinate=water_fragment_1.atoms[0].coordinate,
                                                      pot_derivative_order=1), ref_potential)
        # Second order derivative
        ref_potential = 0
        for frag_atoms in water_fragment_1.atoms:
            ref_potential += frag_atoms.potential(coordinate=water_fragment_2.atoms[0].coordinate,
                                                  pot_derivative_order=1, origin_derivative_order=1)
        assert np.allclose(water_fragment_1.potential(coordinate=water_fragment_2.atoms[0].coordinate,
                                                      pot_derivative_order=1, origin_derivative_order=1), ref_potential)
        ref_potential = 0
        for frag_atoms in water_fragment_2.atoms:
            ref_potential += frag_atoms.potential(coordinate=water_fragment_1.atoms[0].coordinate,
                                                  pot_derivative_order=1, origin_derivative_order=1)
        assert np.allclose(water_fragment_2.potential(coordinate=water_fragment_1.atoms[0].coordinate,
                                                      pot_derivative_order=1, origin_derivative_order=1), ref_potential)
        ref_potential = 0
        for frag_atoms in water_fragment_1.atoms:
            ref_potential += frag_atoms.potential(coordinate=water_fragment_2.atoms[0].coordinate,
                                                  pot_derivative_order=2, origin_derivative_order=0)
        assert np.allclose(water_fragment_1.potential(coordinate=water_fragment_2.atoms[0].coordinate,
                                                      pot_derivative_order=2, origin_derivative_order=0), ref_potential)
        ref_potential = 0
        for frag_atoms in water_fragment_2.atoms:
            ref_potential += frag_atoms.potential(coordinate=water_fragment_1.atoms[0].coordinate,
                                                  pot_derivative_order=2, origin_derivative_order=0)
        assert np.allclose(water_fragment_2.potential(coordinate=water_fragment_1.atoms[0].coordinate,
                                                      pot_derivative_order=2, origin_derivative_order=0), ref_potential)

class TestQuantumFragment:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Common setup for QuantumFragment tests
        self.index = 1
        self.nuclei = [{'element': 'H', 'coordinate': [0, 0, 0], 'index': 0, 'charge': 1},
                  {'element': 'O', 'coordinate': [1, 1, 1], 'index': 1, 'charge': 8}]
        self.e_density_matrix = density_matrix.DensityMatrix(np.zeros((3, 3)))  # Dummy density matrix
        self.test_fragment = fragment.QuantumFragment(self.index, self.nuclei, self.e_density_matrix)

    def test_initialization(self,
                            water_density,
                            hydrogen_nucleus_in_nuclei_list
                            ):
        water_density = density_matrix.DensityMatrix(density=water_density)
        test_fragment2 = fragment.QuantumFragment(index=1, nuclei=hydrogen_nucleus_in_nuclei_list, name="H2O",
                                                  e_density_matrix=water_density)
        assert self.test_fragment.num_nuclei == 2
        assert test_fragment2._name == "H2O"
        assert len(test_fragment2.nuclei) == 1
        assert isinstance(test_fragment2.nuclei[0], particle.Nucleus)

    def test_density_matrix(self):
        assert np.array_equal(self.test_fragment.density, self.e_density_matrix.density)

