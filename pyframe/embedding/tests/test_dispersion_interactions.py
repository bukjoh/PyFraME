"""Tests PyFraME.embedding.dispersion_interactions.py"""
import pytest
import os

from pyframe.embedding import dispersion_interactions, read_input


def test_compute_dispersion_interactions():
    # Unperturbed repulsion potential (Pauli-Repulsion)
    core_oxygen, env_oxygen = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/two_oxygen_test.json')
    ref_pot = -3.1268623466642326e-05
    assert pytest.approx(ref_pot, abs=1e-12) == dispersion_interactions.compute_dispersion_interactions(core_oxygen,
                                                                                                        env_oxygen)
    core_two_wat, env_two_wat = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/two_wat_test.json')
    ref_pot = -3.60476858172198e-05
    assert pytest.approx(ref_pot, abs=1e-12) == dispersion_interactions.compute_dispersion_interactions(core_two_wat,
                                                                                                        env_two_wat)
