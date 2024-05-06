"""Tests PyFraME.embedding.dispersion_interactions.py"""
import pytest
import numpy as np
import copy

from pyframe.embedding import dispersion_interactions
from pyframe.embedding.pert_tuple_cache import rspPert, rspPertTuple, rspCache


def test_compute_dispersion_interactions(two_oxygen,
                                         two_wat,
                                         neon):
    # Setup
    core_oxygen, env_oxygen = two_oxygen
    core_two_wat, env_two_wat = two_wat
    # Unperturbed dispersion potential
    ref_pot = -3.1268623466642326e-05
    assert (pytest.approx(ref_pot, abs=1e-12) == dispersion_interactions.
            compute_dispersion_interactions(quantum_subsystem=core_oxygen,
                                            classical_subsystem=env_oxygen,
                                            perturbed=False,
                                            method='LJ',
                                            combination_rule='Lorentz-Berthelot'))
    ref_pot = -3.60476858172198e-05
    assert (pytest.approx(ref_pot, abs=1e-12) == dispersion_interactions.
            compute_dispersion_interactions(quantum_subsystem=core_two_wat,
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
    ref_grad = np.array([-1.58864190e-05, -7.65863953e-06, 1.03072146e-09], dtype=np.float64)
    result_geo_and_el_0 = (dispersion_interactions.
                           compute_dispersion_interactions(quantum_subsystem=core_oxygen,
                                                           classical_subsystem=env_oxygen,
                                                           perturbed=True,
                                                           method='LJ',
                                                           combination_rule='Lorentz-Berthelot',
                                                           perturbation_cache=energy_props_geo_and_el0))

    for pert in range(3):
        assert result_geo_and_el_0[perts_geo.h][(pert,)] == pytest.approx(ref_grad[pert], rel=1e-8)
        assert result_geo_and_el_0[perts_el0.h][(pert,)] == 0.0
    # Test if any pert is EL the result becomes 0.0 for every component
    result_geo_el_0 = (dispersion_interactions.
                       compute_dispersion_interactions(quantum_subsystem=core_oxygen,
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
    ref_grad = np.array([[-1.72766238e-05, -8.38523204e-06, 9.76741783e-10],
                         [-1.22100117e-06, -5.54637193e-07, 3.11569837e-10],
                         [-2.05921679e-07, -6.39300312e-08, -3.83459454e-12]])
    result_geo_and_el_0 = (dispersion_interactions.
                           compute_dispersion_interactions(quantum_subsystem=core_two_wat,
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
    result_geo_el_0 = (dispersion_interactions.
                       compute_dispersion_interactions(quantum_subsystem=core_two_wat,
                                                       classical_subsystem=core_two_wat,
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
    ref_contr = [-69.87799665580887, 0.0, 0.0]
    result_g = (dispersion_interactions.compute_dispersion_interactions(quantum_subsystem=core_neon,
                                                                        classical_subsystem=env_neon,
                                                                        perturbed=True,
                                                                        method='LJ',
                                                                        combination_rule='Lorentz-Berthelot',
                                                                        perturbation_cache=energy_props_g))
    for i in range(3):
        assert pytest.approx(ref_contr[i], rel=1e-8) == result_g[perts_g.h][(i,)]
    result_gg = (dispersion_interactions.compute_dispersion_interactions(quantum_subsystem=core_neon,
                                                                         classical_subsystem=env_neon,
                                                                         perturbed=True,
                                                                         method='LJ',
                                                                         combination_rule='Lorentz-Berthelot',
                                                                         perturbation_cache=energy_props_gg))
    ref_contr = [-235.3135504147008, 0.0, 0.0, 33.6162214878144, 0.0, 33.6162214878144]
    counter = 0
    for i in range(3):
        for j in range(i, 3):
            assert pytest.approx(ref_contr[counter], rel=1e-8) == result_gg[perts_gg.h][(i, j)]
            counter += 1
    result_ggg = (dispersion_interactions.compute_dispersion_interactions(quantum_subsystem=core_neon,
                                                                          classical_subsystem=env_neon,
                                                                          perturbed=True,
                                                                          method='LJ',
                                                                          combination_rule='Lorentz-Berthelot',
                                                                          perturbation_cache=energy_props_ggg))
    ref_contr = [-905.6186849531825, 0.0, 0.0, 129.37409785045463, 0.0, 129.37409785045463, 0.0, 0.0, 0.0, 0.0]
    counter = 0
    for i in range(3):
        for j in range(i, 3):
            for k in range(j, 3):
                # print(((i, j, k),))
                # print(result_ggg[perts_ggg.h][(i, j, k)])
                assert pytest.approx(ref_contr[counter], rel=1e-8) == result_ggg[perts_ggg.h][(i, j, k)]
                counter += 1
    result_gggg = (dispersion_interactions.compute_dispersion_interactions(quantum_subsystem=core_neon,
                                                                           classical_subsystem=env_neon,
                                                                           perturbed=True,
                                                                           method='LJ',
                                                                           combination_rule='Lorentz-Berthelot',
                                                                           perturbation_cache=energy_props_gggg))
    ref_contr = [-3920.995417507122, 0.0, 0.0, 560.1422025010173, 0.0, 560.1422025010173, 0.0, 0.0, 0.0, 0.0,
                 -186.71406750033913, 0.0, -62.23802250011304, 0.0, -186.71406750033913]
    counter = 0
    for i in range(3):
        for j in range(i, 3):
            for k in range(j, 3):
                for l in range(k, 3):
                    assert pytest.approx(ref_contr[counter], rel=1e-8) == result_gggg[perts_gggg.h][(i, j, k, l)]
                    counter += 1


def test_compute_dispersion_interactions_gradient(two_oxygen,
                                                  two_wat):
    # Setup
    core_oxygen, env_oxygen = two_oxygen
    core_two_wat, env_two_wat = two_wat
    # Test LJ repulsion gradient
    ref_grad = np.array([-1.58864190e-05, -7.65863953e-06, 1.03072146e-09], dtype=np.float64)
    assert np.allclose(ref_grad, dispersion_interactions.
                       compute_dispersion_interactions_gradient(quantum_subsystem=core_oxygen,
                                                                classical_subsystem=env_oxygen,
                                                                method='LJ',
                                                                combination_rule='Lorentz-Berthelot'))
    ref_grad = np.array([[-1.72766238e-05, -8.38523204e-06, 9.76741783e-10],
                         [-1.22100117e-06, -5.54637193e-07, 3.11569837e-10],
                         [-2.05921679e-07, -6.39300312e-08, -3.83459454e-12]])
    assert np.allclose(ref_grad, dispersion_interactions.
                       compute_dispersion_interactions_gradient(quantum_subsystem=core_two_wat,
                                                                classical_subsystem=env_two_wat,
                                                                method='LJ',
                                                                combination_rule='Lorentz-Berthelot'))
