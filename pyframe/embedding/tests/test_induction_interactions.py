"""Tests PyFraME.embedding.induction_interactions.py"""
import pytest
import numpy as np
import copy

from pyframe.embedding import induction_interactions
from pyframe.embedding.pert_tuple_cache import rspPert, rspPertTuple, rspCache


def test_ind_fock_matrix_contributions(wat_wat,
                                       wat_wat_h_ind,
                                       wat_wat_ind_dip,
                                       acrolein_wat,
                                       acrolein_wat_h_ind,
                                       acrolein_wat_ind_dip,
                                       dummy_integral_driver_factory):
    env_wat = wat_wat[1]
    env_wat.induced_dipoles.induced_dipoles = wat_wat_ind_dip
    driver = dummy_integral_driver_factory(wat_wat_h_ind)
    h_ind = induction_interactions.ind_fock_matrix_contributions(classical_subsystem=env_wat,
                                                                 integral_driver=driver)
    assert h_ind.shape == (24, 24)
    assert np.allclose(h_ind, wat_wat_h_ind)
    assert isinstance(h_ind, np.ndarray)
    env_acrolein = acrolein_wat[1]
    env_acrolein.induced_dipoles.induced_dipoles = acrolein_wat_ind_dip
    driver = dummy_integral_driver_factory(acrolein_wat_h_ind)
    h_ind = induction_interactions.ind_fock_matrix_contributions(classical_subsystem=env_acrolein,
                                                                 integral_driver=driver)
    assert h_ind.shape == (24, 24)
    assert np.allclose(h_ind, acrolein_wat_h_ind)
    assert isinstance(h_ind, np.ndarray)


def test_compute_induction_energy(wat_wat,
                                  wat_wat_ind_dip,
                                  acrolein_wat,
                                  acrolein_wat_ind_dip
                                  ):
    env_wat = wat_wat[1]
    induced_dipoles_wat = -1.0 * wat_wat_ind_dip
    nuclear_fields = -1.0 * np.array([[-0.199083, -0.47195685, -0.06765254],
                                      [-0.19851923, -0.20346855, -0.05862587],
                                      [-0.09091193, -0.30552953, 0.06793896],
                                      [0.1476691, -0.03348313, 0.382633],
                                      [0.13236691, -0.06417787, 0.20803882],
                                      [0.29853772, -0.21678522, 0.84407775],
                                      [-0.27527766, 0.33289472, 0.09254776],
                                      [-0.16068392, 0.18100274, 0.11845155],
                                      [-0.14574911, 0.26154579, -0.00559795],
                                      [0.14652788, 0.03254194, -0.31427559],
                                      [0.37533683, 0.10413035, -0.86909412],
                                      [0.10517092, -0.05743362, -0.2370408]])
    electric_fields = -1.0 * np.array([[0.19711192, 0.43265575, 0.05777407],
                                       [0.19064269, 0.19379572, 0.05348263],
                                       [0.09470435, 0.29246066, -0.06234231],
                                       [-0.14721846, 0.03552778, -0.39864684],
                                       [-0.13360034, 0.06732875, -0.21512317],
                                       [-0.29680713, 0.23716673, -0.88827068],
                                       [0.25810129, -0.30820227, -0.07922521],
                                       [0.15757219, -0.17490486, -0.10992855],
                                       [0.14221186, -0.24626534, 0.00491036],
                                       [-0.14723334, -0.03442903, 0.32690342],
                                       [-0.38115301, -0.11381754, 0.91989704],
                                       [-0.10469259, 0.06033117, 0.24452343]])
    e_ind = induction_interactions.compute_induction_energy(induced_dipoles=induced_dipoles_wat,
                                                            total_fields=(nuclear_fields + electric_fields
                                                                          + env_wat.multipole_fields))
    e_nuc_ind = induction_interactions.compute_induction_energy(induced_dipoles=induced_dipoles_wat,
                                                                total_fields=nuclear_fields)
    e_mul_ind = induction_interactions.compute_induction_energy(induced_dipoles=induced_dipoles_wat,
                                                                total_fields=env_wat.multipole_fields)
    e_el_ind = induction_interactions.compute_induction_energy(induced_dipoles=induced_dipoles_wat,
                                                               total_fields=electric_fields)
    assert e_ind == pytest.approx(-0.01006509940518837, abs=1e-8)
    assert e_nuc_ind == pytest.approx(-0.006052391164318287, abs=1e-8)
    assert e_mul_ind == pytest.approx(-0.0009277338262916619, abs=1e-8)
    assert e_el_ind == pytest.approx(-0.003084974414578435, abs=1e-8)
    assert isinstance(e_ind, float)
    env_acrolein = acrolein_wat[1]
    induced_dipoles_acrolein = -1.0 * acrolein_wat_ind_dip
    nuclear_fields = -1.0 * np.array([[6.12879109e-01, -1.55417637e-02, 1.73067342e-04],
                                      [1.04301145e+00, -1.60325172e-01, 7.96009519e-05],
                                      [4.11031628e-01, -7.55901764e-02, 1.35767536e-04],
                                      [7.86367117e-01, -6.53785251e-02, 1.55895494e-04],
                                      [5.02156493e-01, -5.55531558e-02, 1.55060559e-04],
                                      [-3.19153416e-01, -5.93251403e-01, 2.22276122e-04],
                                      [-4.37631143e-01, -9.92073947e-01, 6.02737985e-04],
                                      [-1.66359890e-01, -4.25705483e-01, -4.61888783e-04],
                                      [-3.64444960e-01, -7.53873295e-01, 3.52231995e-04],
                                      [-2.31122829e-01, -5.02872399e-01, -2.33641946e-04]])
    electric_fields = -1.0 * np.array([[-6.16582892e-01, 1.07395149e-02, -1.75560439e-04],
                                       [-1.05955087e+00, 1.51020748e-01, -8.55583977e-05],
                                       [-4.14238593e-01, 7.37808927e-02, -1.37740454e-04],
                                       [-7.93975504e-01, 5.85703570e-02, -1.60027257e-04],
                                       [-5.05833874e-01, 5.25414026e-02, -1.57389039e-04],
                                       [3.26147215e-01, 5.92319398e-01, -2.25213665e-04],
                                       [4.56788209e-01, 9.99796691e-01, -6.18164979e-04],
                                       [1.70121056e-01, 4.26930308e-01, 4.65778525e-04],
                                       [3.75860864e-01, 7.55273447e-01, -3.58332160e-04],
                                       [2.36323858e-01, 5.03507098e-01, 2.34844522e-04]])
    e_ind = induction_interactions.compute_induction_energy(induced_dipoles=induced_dipoles_acrolein,
                                                            total_fields=(nuclear_fields + electric_fields
                                                                          + env_acrolein.multipole_fields))
    e_nuc_ind = induction_interactions.compute_induction_energy(induced_dipoles=induced_dipoles_acrolein,
                                                                total_fields=nuclear_fields)
    e_mul_ind = induction_interactions.compute_induction_energy(induced_dipoles=induced_dipoles_acrolein,
                                                                total_fields=env_acrolein.multipole_fields)
    e_el_ind = induction_interactions.compute_induction_energy(induced_dipoles=induced_dipoles_acrolein,
                                                               total_fields=electric_fields)
    assert e_ind == pytest.approx(-0.000751929428, abs=1e-9)
    assert e_nuc_ind == pytest.approx(0.042124675086, abs=1e-8)
    assert e_mul_ind == pytest.approx(0.000114734186, abs=1e-10)
    assert e_el_ind == pytest.approx(-0.0429913387, abs=1e-8)
    assert isinstance(e_ind, float)


def test_compute_induction_energy_gradient(two_oxygen, two_wat):
    # Setup
    core_oxygen, env_oxygen = two_oxygen
    core_two_wat, env_two_wat = two_wat
    # Calculate induced dipoles
    env_oxygen.solve_induced_dipoles(external_fields=core_oxygen.compute_nuclear_fields(env_oxygen.coordinates))
    env_two_wat.solve_induced_dipoles(external_fields=core_two_wat.compute_nuclear_fields(env_two_wat.coordinates))
    # Calculate nuclear field gradients
    nuclear_field_gradients_two_ox = core_oxygen.compute_nuclear_field_gradients(env_oxygen.coordinates)
    nuclear_field_gradients_two_wat = core_two_wat.compute_nuclear_field_gradients(env_two_wat.coordinates)
    # TODO missing electric field gradient
    # Calculate induction energy gradient
    induction_energy_gradient_two_ox = induction_interactions.compute_induction_energy_gradient(
        induced_dipoles=env_oxygen.induced_dipoles.induced_dipoles,
        total_field_gradients=nuclear_field_gradients_two_ox)
    induction_energy_gradient_two_wat = induction_interactions.compute_induction_energy_gradient(
        induced_dipoles=env_two_wat.induced_dipoles.induced_dipoles,
        total_field_gradients=nuclear_field_gradients_two_wat)
    ref_energy_gradient_two_ox = np.array([1.49189868e-03, 9.47675498e-04, 1.62073790e-06], dtype=np.float64)
    ref_energy_gradient_two_wat = np.array([[4.85319304e-03, 2.74198596e-03, 1.12304565e-06],
                                            [1.16342801e-03, 6.03914172e-04, -1.47528824e-07],
                                            [5.31607331e-04, 1.47436487e-04, 1.78351047e-07]], dtype=np.float64)
    assert np.allclose(ref_energy_gradient_two_ox, induction_energy_gradient_two_ox)
    assert np.allclose(ref_energy_gradient_two_wat, induction_energy_gradient_two_wat)


def test_compute_rsp_induction_energy(wat_wat):
    # Setup
    core, env = wat_wat
    geo_templ = rspPert('GEO', 0.0)
    el_0_templ = rspPert('EL', 0.0)
    el_0_4_templ = rspPert('EL', 0.4)
    perts_geo_el0 = rspPertTuple([copy.deepcopy(geo_templ), copy.deepcopy(el_0_templ)])
    comps_geo_el0 = {((4, 1),), ((5, 2),), ((10, 0),)}
    p_tuple_geo_el0 = [perts_geo_el0]
    energy_props_geo_el0 = rspCache(p_tuple_geo_el0, k=1, n=0, comps=comps_geo_el0)
    # Important set Ids!
    energy_props_geo_el0.setIds()
    density_bank = dict()
    # induction_interactions.compute_rsp_induction_energy(input_cache=energy_props_geo_el0,
    #                                                     density_bank=density_bank,
    #                                                     quantum_subsystem=core,
    #                                                     classical_subsystem=env)
