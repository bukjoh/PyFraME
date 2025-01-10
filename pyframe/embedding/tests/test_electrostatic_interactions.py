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
                                                                                     integral_driver=driver)
    assert e_nuc_es == pytest.approx(ref_energy, 1e-9)
    assert f_el_es == pytest.approx(es_fock_contr, 1e-9)
    core, env = wat_wat
    driver = dummy_integral_driver_factory(wat_wat_es_fock_contr)
    e_nuc_es, f_el_es = electrostatic_interactions.compute_electrostatic_interaction(quantum_subsystem=core,
                                                                                     classical_subsystem=env,
                                                                                     integral_driver=driver)
    D = wat_wat_density
    e_el_es = np.einsum("ab, ab", D, f_el_es)
    assert e_nuc_es == pytest.approx(-0.08545956, abs=1.5e-8)
    assert e_el_es == pytest.approx(0.00987780, abs=1.5e-8)


def test_compute_electrostatic_nuclear_energy(wat_wat,
                                              butadiene_water,
                                              two_oxygen):
    core_wat_wat, env_wat_wat = wat_wat
    e_nuc_es = electrostatic_interactions.compute_electrostatic_nuclear_energy(quantum_subsystem=core_wat_wat,
                                                                               classical_subsystem=env_wat_wat)
    assert e_nuc_es == pytest.approx(-0.08545956, abs=1.5e-8)

    core_but_wat, env_but_wat = butadiene_water
    e_nuc_es = electrostatic_interactions.compute_electrostatic_nuclear_energy(quantum_subsystem=core_but_wat,
                                                                               classical_subsystem=env_but_wat)
    assert e_nuc_es == pytest.approx(0.16763472778701938, abs=1.5e-8)
    core_ox, env_ox = two_oxygen
    e_nuc_es = electrostatic_interactions.compute_electrostatic_nuclear_energy(quantum_subsystem=core_ox,
                                                                               classical_subsystem=env_ox)
    assert e_nuc_es == pytest.approx(-0.5418862937289033, abs=1.5e-8)


def test_compute_electrostatic_nuclear_gradients(neon, wat_wat, butadiene_water):
    core_ne, env_ne = neon

    ref_energy = [-2.3142853266046646, 0.0, 0.0]
    gradient = electrostatic_interactions.compute_electrostatic_nuclear_gradients(quantum_subsystem=core_ne,
                                                                                  classical_subsystem=env_ne)
    assert np.allclose(ref_energy, gradient)
    core_wat, env_wat = wat_wat
    ref_energy = np.array([[-0.21247583, 0.03107111, 0.04910824],
                           [-0.02815794, -0.05766402, -0.02072039],
                           [-0.04120633, 0.03522868, 0.020967]])
    gradient = electrostatic_interactions.compute_electrostatic_nuclear_gradients(quantum_subsystem=core_wat,
                                                                                  classical_subsystem=env_wat)
    assert np.allclose(ref_energy, gradient)
    core_but, env_but = butadiene_water
    ref_energy = np.array([[-0.00719841, 0.01379588, -0.00045358],
                           [-0.01500085, -0.00583608, -0.00916102],
                           [-0.00756524, -0.0459053, -0.04311232],
                           [0.01163024, -0.04039575, -0.02034383],
                           [-0.00062134, 0.00199819, 0.00061198],
                           [0.0002471, 0.00215507, 0.00084395],
                           [-0.00072547, 0.00034058, 0.00118228],
                           [-0.00446107, -0.00829789, -0.00891585],
                           [0.00185617, -0.00390107, -0.00197242],
                           [0.00134642, -0.00484623, -0.00165279]])
    gradient = electrostatic_interactions.compute_electrostatic_nuclear_gradients(quantum_subsystem=core_but,
                                                                                  classical_subsystem=env_but)
    assert np.allclose(ref_energy, gradient)


def test_compute_perturbed_electrostatic_interaction(neon):
    core_ne, env_ne = neon
    ref_energy = [-2.3142853266046646, 0.0, 0.0]
    for i in range(3):
        perturbation = [0, 0, 0]
        perturbation[i] += 1
        assert (electrostatic_interactions.
                compute_perturbed_electrostatic_interaction(quantum_subsystem=core_ne,
                                                            classical_subsystem=env_ne,
                                                            perturbation_indices=perturbation,
                                                            nucleus_idx=0) == pytest.approx(ref_energy[i],
                                                                                            rel=1e-8))
    ref_energy = [-2.2266673873992375, 0.0, 0.0, 1.1133336936996188, 0.0, 1.1133336936996188]
    counter = 0
    for i in range(3):
        perturbation1 = [0, 0, 0]
        perturbation1[i] += 1
        for j in range(i, 3):
            perturbation2 = perturbation1
            perturbation2[j] += 1
            assert (electrostatic_interactions.
                    compute_perturbed_electrostatic_interaction(quantum_subsystem=core_ne,
                                                                classical_subsystem=env_ne,
                                                                perturbation_indices=perturbation2,
                                                                nucleus_idx=0) == pytest.approx(ref_energy[counter],
                                                                                                rel=1e-8))
            counter += 1
