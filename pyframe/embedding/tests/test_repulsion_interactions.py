"""Tests PyFraME.embedding.repulsion_interactions.py"""
import pytest
import numpy as np
import copy

from pyframe.embedding import repulsion_interactions
from pyframe.embedding.pert_tuple_cache import rspPert, rspPertTuple, rspCache


def test_compute_repulsion_interactions(two_oxygen,
                                        two_wat,
                                        neon):
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
        assert result_geo_and_el_0[perts_geo.h][(pert,)] == pytest.approx(ref_grad[pert // 3, pert % 3], rel=1e-8)
        assert result_geo_and_el_0[perts_el0.h][(pert,)] == 0.0
    # Test if any pert is EL the result becomes 0.0 for every component
    comps_geo_el0 = {((0, 0),), ((1, 1),), ((2, 2),),
                     ((3, 0),), ((4, 1),), ((5, 2),),
                     ((6, 0),), ((7, 1),), ((8, 2),)}
    energy_props_geo_el0 = rspCache(p_tuple_geo_el0, k=1, n=0, comps=comps_geo_el0)
    result_geo_el_0 = (repulsion_interactions.
                       compute_repulsion_interactions(quantum_subsystem=core_two_wat,
                                                      classical_subsystem=env_two_wat,
                                                      perturbed=True,
                                                      method='LJ',
                                                      combination_rule='Lorentz-Berthelot',
                                                      perturbation_cache=energy_props_geo_el0))
    for res in result_geo_el_0[perts_geo_el0.h].values():
        assert res == 0.0
    # Test Ne - Ne
    geo_templ = rspPert('GEO', 0.0)
    core_neon, env_neon = neon
    perts_g = rspPertTuple([copy.deepcopy(geo_templ)])
    perts_gg = rspPertTuple([copy.deepcopy(geo_templ),
                             copy.deepcopy(geo_templ)])
    perts_ggg = rspPertTuple([copy.deepcopy(geo_templ),
                              copy.deepcopy(geo_templ),
                              copy.deepcopy(geo_templ)])
    perts_gggg = rspPertTuple([copy.deepcopy(geo_templ),
                               copy.deepcopy(geo_templ),
                               copy.deepcopy(geo_templ),
                               copy.deepcopy(geo_templ)])
    p_tuple_g = [perts_g]
    p_tuple_gg = [perts_gg]
    p_tuple_ggg = [perts_ggg]
    p_tuple_gggg = [perts_gggg]
    comps_g = {((0,),), ((1,),), ((2,),)}
    comps_gg = {((0, 0),), ((1, 1),), ((2, 2),), ((0, 1),), ((0, 2),), ((1, 2),)}
    comps_ggg = {((0, 0, 0),), ((0, 0, 1),), ((0, 0, 2),), ((0, 1, 1),), ((0, 1, 2),), ((0, 2, 2),), ((1, 1, 1),),
                 ((1, 1, 2),), ((1, 2, 2),), ((2, 2, 2),)}
    comps_gggg = {((0, 0, 0, 0),), ((0, 0, 0, 1),), ((0, 0, 0, 2),), ((0, 0, 1, 1),), ((0, 0, 1, 2),), ((0, 0, 2, 2),),
                  ((0, 1, 1, 1),), ((0, 1, 1, 2),), ((0, 1, 2, 2),), ((0, 2, 2, 2),), ((1, 1, 1, 1),), ((1, 1, 1, 2),),
                  ((1, 1, 2, 2),), ((1, 2, 2, 2),), ((2, 2, 2, 2),)}
    energy_props_g = rspCache(p_tuple_g, k=0, n=0, comps=comps_g)
    energy_props_gg = rspCache(p_tuple_gg, k=0, n=0, comps=comps_gg)
    energy_props_ggg = rspCache(p_tuple_ggg, k=0, n=0, comps=comps_ggg)
    energy_props_gggg = rspCache(p_tuple_gggg, k=0, n=0, comps=comps_gggg)
    ref_contr = [422.92289720831826, 0.0, 0.0]
    result_g = (repulsion_interactions.
                compute_repulsion_interactions(quantum_subsystem=core_neon,
                                               classical_subsystem=env_neon,
                                               perturbed=True,
                                               method='LJ',
                                               combination_rule='Lorentz-Berthelot',
                                               perturbation_cache=energy_props_g))
    for i in range(3):
        assert pytest.approx(ref_contr[i], rel=1e-8) == result_g[perts_g.h][(i,)]
    result_gg = (repulsion_interactions.
                 compute_repulsion_interactions(quantum_subsystem=core_neon,
                                                classical_subsystem=env_neon,
                                                perturbed=True,
                                                method='LJ',
                                                combination_rule='Lorentz-Berthelot',
                                                perturbation_cache=energy_props_gg))
    ref_contr = [2644.9228090075026, 0.0, 0.0, -203.45560069288484, 0.0, -203.45560069288484]
    counter = 0
    for i in range(3):
        for j in range(i, 3):
            assert pytest.approx(ref_contr[counter], rel=1e-8) == result_gg[perts_gg.h][(i, j)]
            counter += 1
    result_ggg = (repulsion_interactions.
                  compute_repulsion_interactions(quantum_subsystem=core_neon,
                                                 classical_subsystem=env_neon,
                                                 perturbed=True,
                                                 method='LJ',
                                                 combination_rule='Lorentz-Berthelot',
                                                 perturbation_cache=energy_props_ggg))
    ref_contr = [17813.50944635706, 0.0, 0.0, -1370.2699574120813, 0.0, -1370.2699574120813, 0.0, 0.0, 0.0, 0.0]
    counter = 0
    for i in range(3):
        for j in range(i, 3):
            for k in range(j, 3):
                assert pytest.approx(ref_contr[counter], rel=1e-8) == result_ggg[perts_ggg.h][(i, j, k)]
                counter += 1
    result_gggg = (repulsion_interactions.
                   compute_repulsion_interactions(quantum_subsystem=core_neon,
                                                  classical_subsystem=env_neon,
                                                  perturbed=True,
                                                  method='LJ',
                                                  combination_rule='Lorentz-Berthelot',
                                                  perturbation_cache=energy_props_gggg))
    ref_contr = [128543.22698464915, 0.0, 0.0, -9887.940537280705, 0.0, -9887.940537280705, 0.0, 0.0, 0.0, 0.0,
                 1977.588107456141, 0.0, 659.1960358187137, 0.0, 1977.588107456141]
    counter = 0
    for i in range(3):
        for j in range(i, 3):
            for k in range(j, 3):
                for l in range(k, 3):
                    assert pytest.approx(ref_contr[counter], rel=1e-8) == result_gggg[perts_gggg.h][(i, j, k, l)]
                    counter += 1


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
