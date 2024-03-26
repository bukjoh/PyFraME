"""Tests PyFraME.embedding.repulsion_interactions.py"""
import pytest
import os

from pyframe.embedding import repulsion_interactions, read_input


def test_compute_repulsion_interactions():
    # Unperturbed repulsion potential (Pauli-Repulsion)
    core_oxygen, env_oxygen = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/two_oxygen_test.json')
    ref_pot = 8.976559066274837e-07
    assert pytest.approx(ref_pot, abs=1e-12) == repulsion_interactions.compute_repulsion_interactions(core_oxygen,
                                                                                                      env_oxygen)
    core_two_wat, env_two_wat = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/two_wat_test.json')
    ref_pot = 9.161786856931556e-07
    assert pytest.approx(ref_pot, abs=1e-12) == repulsion_interactions.compute_repulsion_interactions(core_two_wat,
                                                                                                      env_two_wat)
