"""Tests PyFraME.embedding.electrostatic_interactions.py"""
import pytest
import numpy as np

from pyframe.embedding import (polytensor, electrostatic_interactions, fragment, subsystem)


def test_compute_particle_interactions(
        phys_constants,
        hydrogen_atom1,
        hydrogen_atom2,
        hydrogen_atom_dipole_z,
        hydrogen_atom_dipole_y,
        hydrogen_atom_dipole_z_translated_x,
        hydrogen_atom_dipole_z_translated_z,
        hydrogen_atom_dipole_translated,
        hydrogen1_nucleus,
        hydrogen2_nucleus,
        oxygen_atom1,
        oxygen_nucleus
):
    # Test atoms interaction
    # Test Monopole-Monopole interaction
    ref_interaction_energy_monopole = 2.30707755234174E-18
    assert electrostatic_interactions.compute_particle_interactions(hydrogen_atom1, hydrogen_atom2) \
           * phys_constants.hartree2J == pytest.approx(ref_interaction_energy_monopole, 1e-9)
    assert electrostatic_interactions.compute_particle_interactions(hydrogen_atom2, hydrogen_atom1) \
           * phys_constants.hartree2J == pytest.approx(ref_interaction_energy_monopole, 1e-9)
    # Test orthogonal Dipole-Dipole interaction
    ref_interaction_energy_orth_dipole = 0.0
    assert electrostatic_interactions.compute_particle_interactions(hydrogen_atom_dipole_z,
                                                                    hydrogen_atom_dipole_y) \
           * phys_constants.hartree2J == pytest.approx(ref_interaction_energy_orth_dipole, 1e-9)
    # Test parallel Dipole-Dipole interaction
    ref_interaction_energy_parallel_dipole_xz = 6.46047513751174E-19
    assert electrostatic_interactions.compute_particle_interactions(hydrogen_atom_dipole_z,
                                                                    hydrogen_atom_dipole_z_translated_x) \
           * phys_constants.hartree2J == \
           pytest.approx(ref_interaction_energy_parallel_dipole_xz, 1e-9)
    ref_interaction_energy_parallel_dipole_zz = -1.29209502750235E-18
    assert electrostatic_interactions.compute_particle_interactions(hydrogen_atom_dipole_z,
                                                                    hydrogen_atom_dipole_z_translated_z) \
           * phys_constants.hartree2J == \
           pytest.approx(ref_interaction_energy_parallel_dipole_zz, 1e-9)
    # Test multi-directional Dipole-Dipole interaction
    ref_interaction_energy_dipole = 2.63168830505711E-21
    assert electrostatic_interactions.compute_particle_interactions(hydrogen_atom_dipole_z,
                                                                    hydrogen_atom_dipole_translated) \
           * phys_constants.hartree2J == \
           pytest.approx(ref_interaction_energy_dipole, 1e-9)
    # Test nuclei interaction
    assert electrostatic_interactions.compute_particle_interactions(hydrogen1_nucleus, hydrogen2_nucleus) \
           == pytest.approx(0.34335113566, 1e-9)
    assert electrostatic_interactions.compute_particle_interactions(hydrogen1_nucleus, hydrogen2_nucleus) \
           == electrostatic_interactions.compute_particle_interactions(hydrogen2_nucleus, hydrogen1_nucleus)
    # with pytest.raises(ValueError, match="r_a and r_b cannot be equal."):
    #     electrostatic_interactions.compute_particle_interactions(hydrogen1_nucleus, hydrogen1_nucleus)
    assert electrostatic_interactions.compute_particle_interactions(oxygen_nucleus, hydrogen2_nucleus) \
           == pytest.approx(4.39578846868, 1e-9)
    assert electrostatic_interactions.compute_particle_interactions(oxygen_nucleus, hydrogen1_nucleus) \
           == pytest.approx(4.35039628273, 1e-9)
    # Test nucleus atom interaction
    # Test Multipole (rank =3) - Charge interaction
    ref_potential = np.array(
        [0.75202669489040475, 6.3679443173877762e-2, 3.0699045537920477e-2, -4.1315647410858666e-6,
         9.5311929756771267e-3, 7.7985188788112876e-3, -1.0495468203592808e-6,
         -2.8858133904975198e-3, -5.0597310570715463e-7, -6.6453795851796043e-3,
         1.7845230927165381e-3, 2.4879519222560046e-3, -3.3483563607254773e-7,
         -9.6388595705967816e-5, -2.1422168821470319e-7, -1.6881344970105696e-3,
         -1.6741238607204253e-3, 6.2537563711756675e-9, -8.1382806153557845e-4,
         3.2858187970137201e-7], dtype=np.float64)
    ref_polytensor = polytensor.FirstDegreePolytensor(rank=3, tensor_data=ref_potential)
    taylor_coefficient = oxygen_atom1.taylor_coefficients.data
    ref_interaction_energy = ref_polytensor.dot_first_degree(polytensor.FirstDegreePolytensor.
                                                             multiply_elementwise(oxygen_atom1.
                                                                                  multipoles_with_degeneracy,
                                                                                  taylor_coefficient))
    assert electrostatic_interactions.compute_particle_interactions(oxygen_atom1, oxygen_nucleus) \
           == pytest.approx(ref_interaction_energy, 1e-9)

    assert electrostatic_interactions.compute_particle_interactions(oxygen_nucleus, oxygen_atom1) \
           == pytest.approx(ref_interaction_energy, 1e-9)


def test_compute_fragment_particle_interactions(
        phys_constants,
        water_fragment_dict,
        oxygen_nucleus,
        oxygen_atom1
):
    water_fragment = fragment.ClassicalFragment(**water_fragment_dict)
    interaction_energy = electrostatic_interactions.compute_particle_interactions
    # Test fragment nucleus
    es_energy = electrostatic_interactions.compute_fragment_particle_interactions(oxygen_nucleus,
                                                                                  water_fragment)
    ref_energy = interaction_energy(water_fragment.atoms[0], oxygen_nucleus) \
                 + interaction_energy(water_fragment.atoms[1], oxygen_nucleus) \
                 + interaction_energy(water_fragment.atoms[2], oxygen_nucleus)
    assert ref_energy == pytest.approx(es_energy, 1e-9)
    # Test fragment atom
    es_energy = electrostatic_interactions.compute_fragment_particle_interactions(oxygen_atom1,
                                                                                  water_fragment)
    ref_energy = interaction_energy(water_fragment.atoms[0], oxygen_atom1) \
                 + interaction_energy(water_fragment.atoms[1], oxygen_atom1) \
                 + interaction_energy(water_fragment.atoms[2], oxygen_atom1)
    assert ref_energy == pytest.approx(es_energy, 1e-9)


def test_fragments_interaction(
        phys_constants,
        water_fragments
):
    water_fragment_1 = water_fragments[0]
    water_fragment_2 = water_fragments[1]
    interaction_energy = electrostatic_interactions.compute_particle_interactions
    es_energy = electrostatic_interactions.compute_fragment_interactions(water_fragment_1, water_fragment_2)
    ref_energy = 0
    for atoms in water_fragment_2.atoms:
        ref_energy += interaction_energy(water_fragment_1.atoms[0], atoms) \
                      + interaction_energy(water_fragment_1.atoms[1], atoms) \
                      + interaction_energy(water_fragment_1.atoms[2], atoms)
    assert ref_energy == pytest.approx(es_energy, 1e-9)


def test_compute_particle_interactions_invalid_input():
    # Test when compute_particle_interactions is called with invalid particle types
    with pytest.raises(TypeError):
        electrostatic_interactions.compute_particle_interactions(fragment.ClassicalFragment(),
                                                                 fragment.ClassicalFragment())


def test_compute_fragment_interactions_invalid_input():
    # Test when compute_fragment_interactions is called with invalid fragment types
    with pytest.raises(TypeError):
        electrostatic_interactions.compute_fragment_interactions(subsystem.QuantumSubsystem(),
                                                                 fragment.ClassicalFragment())
    with pytest.raises(TypeError):
        electrostatic_interactions.compute_fragment_interactions(fragment.ClassicalFragment(),
                                                                 subsystem.QuantumSubsystem())


def test_compute_electrostatic_interaction_invalid_input():
    # Test when compute_electrostatic_interaction is called with invalid subsystem types
    with pytest.raises(TypeError):
        electrostatic_interactions.compute_electrostatic_interaction(subsystem.QuantumSubsystem(),
                                                                     fragment.ClassicalFragment(),
                                                                     None)


def test_compute_electrostatic_interaction_empty_list(
        act_wat
):
    core = act_wat[0]
    # Test when empty lists are passed to compute_electrostatic_interaction
    e_nuc_es, f_el_es = electrostatic_interactions.compute_electrostatic_interaction(core,
                                                                                     [],
                                                                                     None)
    assert e_nuc_es == 0
    assert f_el_es is None


def test_compute_electrostatic_interaction(
        act_wat_es_fock_contr,
        act_wat,
        wat_wat,
        wat_wat_es_fock_contr,
        wat_wat_density,
        dummy_integral_driver_factory
):
    core, env = act_wat
    driver = dummy_integral_driver_factory(act_wat_es_fock_contr)
    ref_energy = 0
    for nucleus in core.nuclei:
        for fragments in env.classical_fragments:
            ref_energy += electrostatic_interactions.compute_fragment_particle_interactions(nucleus, fragments)
    es_fock_contr = electrostatic_interactions.es_fock_matrix_contributions(env, driver)
    e_nuc_es, f_el_es = electrostatic_interactions.compute_electrostatic_interaction(quantum_subsystem=core,
                                                                                     classical_subsystem=env,
                                                                                     integral_drv=driver)
    assert e_nuc_es == pytest.approx(ref_energy, 1e-9)
    assert f_el_es == pytest.approx(es_fock_contr, 1e-9)
    core, env = wat_wat
    driver = dummy_integral_driver_factory(wat_wat_es_fock_contr)
    e_nuc_es, f_el_es = electrostatic_interactions.compute_electrostatic_interaction(quantum_subsystem=core,
                                                                                     classical_subsystem=env,
                                                                                     integral_drv=driver)
    D = wat_wat_density
    e_el_es = np.einsum("ab, ab", D, f_el_es)
    assert e_nuc_es == pytest.approx(-0.08545956, abs=1.5e-8)
    assert e_el_es == pytest.approx(0.00987780, abs=1.5e-8)


def test_compute_perturbed_electrostatic_interaction(two_oxygen,
                                                     neon):
    # FIXME somehow this is not clear??
    core, env = neon
    distance = core.coordinates[0] - env.coordinates[0]
    from pyframe.embedding import tensor_tools, constants
    # print(tensor_tools.compute_interaction_tensor_element(distance_vector=distance,
    #                                                       multi_index=[np.array([0, 0, 0], dtype=np.int64),
    #                                                                    np.array([1, 0, 0], dtype=np.int64)],
    #                                                       tensor_coefficients=constants.values.tensor_coefficients))
    # print(electrostatic_interactions.compute_perturbed_electrostatic_interaction(quantum_subsystem=core,
    #                                                                              classical_subsystem=env,
    #                                                                              perturbation_indices=[1, 0, 0],
    #                                                                              nucleus_idx=0))
    # in frame: -2.3142853266046650 not in frame: 2.314285326604665
    # for deriv in y and z direction is 0.0 as in the test

    # for second order derivative
    # [-2.2266673873992375_dp, 0.0_dp, 0.0_dp, 1.1133336936996188_dp, 0.0_dp, &
    #                             1.1133336936996188_dp]
    # my results 200 -2.2266673873992375 020 1.1133336936996188 002 1.1133336936996188
