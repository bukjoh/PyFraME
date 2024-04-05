"""Tests PyFraME.embedding.repulsion_interactions.py"""
import pytest
import os
import numpy as np

from pyframe.embedding import repulsion_interactions, read_input


def test_compute_repulsion_interactions(two_oxygen,
                                        two_wat):
    # Setup
    core_oxygen, env_oxygen = two_oxygen
    core_two_wat, env_two_wat = two_wat
    # Unperturbed repulsion potential (Pauli-Repulsion)
    ref_pot = 8.976559066274837e-07
    assert pytest.approx(ref_pot, abs=1e-12) == repulsion_interactions.compute_repulsion_interactions(core_oxygen,
                                                                                                      env_oxygen)
    ref_pot = 9.161786856931556e-07
    assert pytest.approx(ref_pot, abs=1e-12) == repulsion_interactions.compute_repulsion_interactions(core_two_wat,
                                                                                                      env_two_wat)
    # Test LJ repulsion gradient
    ref_grad = np.array([9.12130838e-07, 4.39726617e-07, -5.91796568e-11], dtype=np.float64)
    assert np.allclose(ref_grad, repulsion_interactions.compute_repulsion_interactions(core_oxygen,
                                                                                       env_oxygen,
                                                                                       1))
    ref_grad = np.array([[9.23383377e-07, 4.45393614e-07, -5.99020897e-11],
                         [1.09111035e-08, 4.95471117e-09, -2.78556145e-12],
                         [3.85015244e-10, 1.19512076e-10, 6.99262163e-15]])
    assert np.allclose(ref_grad, repulsion_interactions.compute_repulsion_interactions(core_two_wat,
                                                                                       env_two_wat,
                                                                                       1))
