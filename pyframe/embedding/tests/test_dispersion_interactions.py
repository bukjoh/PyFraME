"""Tests PyFraME.embedding.dispersion_interactions.py"""
import pytest
import os
import numpy as np

from pyframe.embedding import dispersion_interactions, read_input


def test_compute_dispersion_interactions(two_oxygen,
                                         two_wat):
    # Setup
    core_oxygen, env_oxygen = two_oxygen
    core_two_wat, env_two_wat = two_wat
    # Unperturbed dispersion potential
    ref_pot = -3.1268623466642326e-05
    assert pytest.approx(ref_pot, abs=1e-12) == dispersion_interactions.compute_dispersion_interactions(core_oxygen,
                                                                                                        env_oxygen)
    ref_pot = -3.60476858172198e-05
    assert pytest.approx(ref_pot, abs=1e-12) == dispersion_interactions.compute_dispersion_interactions(core_two_wat,
                                                                                                        env_two_wat)
    # Test LJ dispersion gradient
    ref_grad = np.array([-1.58864190e-05, -7.65863953e-06, 1.03072146e-09], dtype=np.float64)
    assert np.allclose(ref_grad, dispersion_interactions.compute_dispersion_interactions(core_oxygen,
                                                                                         env_oxygen,
                                                                                         1))
    ref_grad = np.array([[-1.72766238e-05, -8.38523204e-06, 9.76741783e-10],
                         [-1.22100117e-06, -5.54637193e-07, 3.11569837e-10],
                         [-2.05921679e-07, -6.39300312e-08, -3.83459454e-12]])
    assert np.allclose(ref_grad, dispersion_interactions.compute_dispersion_interactions(core_two_wat,
                                                                                         env_two_wat,
                                                                                         1))
