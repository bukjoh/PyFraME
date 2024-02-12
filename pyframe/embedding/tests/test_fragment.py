"""Tests PyFraME.embedding.fragment.py"""
import numpy as np
from pyframe.embedding import fragment, density_matrix

def test_init_fragment():
    fragment.Fragment(index=1, name="H2O")


def test_init_classical_fragment(oxygen_atom_in_atoms_list):
    fragment.ClassicalFragment(index=1, atoms=oxygen_atom_in_atoms_list, name="H2O")


def test_init_quantum_fragment(water_density, hydrogen_nucleus_in_nuclei_list):
    water_density = density_matrix.DensityMatrix(density=water_density)
    fragment.QuantumFragment(index=1, nuclei=hydrogen_nucleus_in_nuclei_list, name="H2O", e_density_matrix=water_density)


def test_classical_fragment_potential(water_fragments):
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
        ref_potential += frag_atoms.potential(coordinate=water_fragment_2.atoms[0].coordinate, pot_derivative_order=1)
    assert np.allclose(water_fragment_1.potential(coordinate=water_fragment_2.atoms[0].coordinate,
                                                  pot_derivative_order=1), ref_potential)
    ref_potential = 0
    for frag_atoms in water_fragment_2.atoms:
        ref_potential += frag_atoms.potential(coordinate=water_fragment_1.atoms[0].coordinate, pot_derivative_order=1)
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
