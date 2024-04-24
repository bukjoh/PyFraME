"""Tests PyFraME.embedding.repulsion_interactions.py"""
import pytest
import numpy as np
import copy

from pyframe.embedding import repulsion_interactions, read_input
from pyframe.embedding.pert_tuple_cache import rspPert, rspPertTuple, rspCache


def test_compute_repulsion_interactions(two_oxygen,
                                        two_wat):
    # Setup
    core_oxygen, env_oxygen = two_oxygen
    core_two_wat, env_two_wat = two_wat
    # Unperturbed repulsion potential (Pauli-Repulsion)
    # Test Oxygen - Oxygen
    ref_pot = 8.976559066274837e-07
    assert (pytest.approx(ref_pot, abs=1e-12) == repulsion_interactions.
            compute_repulsion_interactions(quantum_subsystem=core_oxygen,
                                           classical_subsystem=env_oxygen,
                                           perturbed=False,
                                           method='LJ',
                                           combination_rule='Lorentz-Berthelot'))
    # Test Water - Water
    ref_pot = 9.161786856931556e-07
    assert (pytest.approx(ref_pot, abs=1e-12) == repulsion_interactions.
            compute_repulsion_interactions(quantum_subsystem=core_two_wat,
                                           classical_subsystem=env_two_wat,
                                           perturbed=False,
                                           method='LJ',
                                           combination_rule='Lorentz-Berthelot'))
    # First order perturbed repulsion potential
    geo_templ = rspPert('GEO', 0.0)
    el_0_templ = rspPert('EL', 0.0)
    perts_el0 = rspPertTuple([copy.deepcopy(el_0_templ)])
    perts_geo = rspPertTuple([copy.deepcopy(geo_templ)])
    perts_geo_el0 = rspPertTuple([copy.deepcopy(geo_templ), copy.deepcopy(el_0_templ)])
    comps_geo_el0 = {((0, 0),), ((1, 1),), ((2, 2),)}
    comps_geo_and_el0 = {((0,), (0,)), ((1,), (1,)), ((2,), (2,))}
    p_tuple_geo_el0 = [perts_geo_el0]
    p_tuple_geo_and_el0 = [perts_geo, perts_el0]
    energy_props_geo_el0 = rspCache(p_tuple_geo_el0, k=1, n=0, comps=comps_geo_el0)
    energy_props_geo_and_el0 = rspCache(p_tuple_geo_and_el0, k=1, n=0, comps=comps_geo_and_el0)





    # Test Oxygen - Oxygen
    ref_grad = np.array([9.12130838e-07, 4.39726617e-07, -5.91796568e-11])
    result_geo_and_el_0 = (repulsion_interactions.
                           compute_repulsion_interactions(quantum_subsystem=core_oxygen,
                                                          classical_subsystem=env_oxygen,
                                                          perturbed=True,
                                                          method='LJ',
                                                          combination_rule='Lorentz-Berthelot',
                                                          perturbation_cache=energy_props_geo_and_el0))
    for pert in range(3):
        assert result_geo_and_el_0[perts_geo.h][(pert,)] == pytest.approx(ref_grad[pert], rel=1e-8)
        assert result_geo_and_el_0[perts_el0.h][(pert,)] == 0.0
    # Test if any pert is EL the result becomes 0.0 for every component
    result_geo_el_0 = (repulsion_interactions.
                       compute_repulsion_interactions(quantum_subsystem=core_oxygen,
                                                      classical_subsystem=env_oxygen,
                                                      perturbed=True,
                                                      method='LJ',
                                                      combination_rule='Lorentz-Berthelot',
                                                      perturbation_cache=energy_props_geo_el0))
    for res in result_geo_el_0[perts_geo_el0.h].values():
        assert res == 0.0
    # Test Water - Water
    comps_geo_and_el0 = {((0,), (0,)), ((1,), (1,)), ((2,), (2,)),
                         ((3,), (3,)), ((4,), (4,)), ((5,), (5,)),
                         ((6,), (6,)), ((7,), (7,)), ((8,), (8,))}
    energy_props_geo_and_el0 = rspCache(p_tuple_geo_and_el0, k=1, n=0, comps=comps_geo_and_el0)
    ref_grad = np.array([[9.23383377e-07, 4.45393614e-07, -5.99020897e-11],
                         [1.09111035e-08, 4.95471117e-09, -2.78556145e-12],
                         [3.85015244e-10, 1.19512076e-10, 6.99262163e-15]])
    result_geo_and_el_0 = (repulsion_interactions.
                           compute_repulsion_interactions(quantum_subsystem=core_two_wat,
                                                          classical_subsystem=env_two_wat,
                                                          perturbed=True,
                                                          method='LJ',
                                                          combination_rule='Lorentz-Berthelot',
                                                          perturbation_cache=energy_props_geo_and_el0))
    for pert in range(9):
        assert result_geo_and_el_0[perts_geo.h][(pert,)] == pytest.approx(ref_grad[pert//3, pert%3], rel=1e-8)
        assert result_geo_and_el_0[perts_el0.h][(pert,)] == 0.0
    # Test if any pert is EL the result becomes 0.0 for every component
    comps_geo_el0 = {((0, 0),), ((1, 1),), ((2, 2),),
                     ((3, 0),), ((4, 1),), ((5, 2),),
                     ((6, 0),), ((7, 1),), ((8, 2),)}
    energy_props_geo_el0 = rspCache(p_tuple_geo_el0, k=1, n=0, comps=comps_geo_el0)
    result_geo_el_0 = (repulsion_interactions.
                       compute_repulsion_interactions(quantum_subsystem=core_oxygen,
                                                      classical_subsystem=env_oxygen,
                                                      perturbed=True,
                                                      method='LJ',
                                                      combination_rule='Lorentz-Berthelot',
                                                      perturbation_cache=energy_props_geo_el0))
    for res in result_geo_el_0[perts_geo_el0.h].values():
        assert res == 0.0


def test_compute_repulsion_interactions_gradient(two_oxygen,
                                                 two_wat):
    # Setup
    core_oxygen, env_oxygen = two_oxygen
    core_two_wat, env_two_wat = two_wat
    # Test LJ repulsion gradient
    ref_grad = np.array([9.12130838e-07, 4.39726617e-07, -5.91796568e-11], dtype=np.float64)
    assert np.allclose(ref_grad, repulsion_interactions.
                       compute_repulsion_interactions_gradient(quantum_subsystem=core_oxygen,
                                                               classical_subsystem=env_oxygen,
                                                               method='LJ',
                                                               combination_rule='Lorentz-Berthelot'))
    ref_grad = np.array([[9.23383377e-07, 4.45393614e-07, -5.99020897e-11],
                         [1.09111035e-08, 4.95471117e-09, -2.78556145e-12],
                         [3.85015244e-10, 1.19512076e-10, 6.99262163e-15]])
    assert np.allclose(ref_grad, repulsion_interactions.
                       compute_repulsion_interactions_gradient(quantum_subsystem=core_two_wat,
                                                               classical_subsystem=env_two_wat,
                                                               method='LJ',
                                                               combination_rule='Lorentz-Berthelot'))
