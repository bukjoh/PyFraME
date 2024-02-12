"""Tests PyFraME.embedding.electrostatic_interactions.py"""
import json
import os
import pytest
import numpy as np
import scipy
import veloxchem as vlx
from pyframe.embedding import (polytensor, electrostatic_interactions,fragment, vlx_interface, read_input)


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
    assert electrostatic_interactions.compute_particle_interactions(hydrogen1_nucleus, hydrogen1_nucleus) \
           == float('inf')
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


def test_compute_electrostatic_interaction():
    # Internal test
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_test.json')

    act_xyz = """10
    atc
    C                    30.101                    29.705                    29.43
    C                    30.889                    29.91                     30.735
    C                    28.635                    30.016                    29.419
    O                    30.67                     29.421                    28.396
    H                    31.182                    30.941                    30.734
    H                    30.307                    29.618                    31.604
    H                    31.868                    29.391                    30.755
    H                    28.215                    30.575                    30.327
    H                    28.132                    29.059                    29.463
    H                    28.339                    30.503                    28.446
    """
    driver = vlx_interface.EmbeddingIntegralDriver(act_xyz, 'sto-3g')
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

    # Test against echembook
    h2o_xyz = """3
    water
    O        0.0000000000      0.0000000000      0.0000000000                 
    H        0.6891400000      0.8324710000      0.0000000000                 
    H        0.7224340000     -0.8726890000      0.0000000000
    """
    molecule = vlx.Molecule.read_xyz_string(h2o_xyz)
    basis = vlx.MolecularBasis.read(molecule, "cc-pvdz")
    nocc = molecule.number_of_alpha_electrons()
    V_nuc = molecule.nuclear_repulsion_energy()
    # overlap
    overlap_drv = vlx.OverlapIntegralsDriver()
    S = overlap_drv.compute(molecule, basis).to_numpy()
    # kinetic energy
    kinetic_drv = vlx.KineticEnergyIntegralsDriver()
    T = kinetic_drv.compute(molecule, basis).to_numpy()
    # nuclear attraction
    nucpot_drv = vlx.NuclearPotentialIntegralsDriver()
    V = -1.0 * nucpot_drv.compute(molecule, basis).to_numpy()
    # one-electron Hamiltonian
    h = T + V
    # two-electron Hamiltonian
    eri_drv = vlx.ElectronRepulsionIntegralsDriver()
    g = eri_drv.compute_in_memory(molecule, basis)
    # initial guess
    epsilon, C = scipy.linalg.eigh(h, S)
    E_HF, C_HF = vlx_interface.scf_solver(h=h, V_nuc=V_nuc, C=C, nocc=nocc, g=g, S=S)
    # define core env
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/wat_in_wat_test.json')

    #env = subsystem.ClassicalSubsystem(name="4x H2O atoms",
    #                                   input_data=f'{os.path.dirname(__file__)}/data/wat_in_wat_test.json')
    #core = subsystem.QuantumSubsystem(name="1x H2O",
    #                                  input_data=f'{os.path.dirname(__file__)}/data/wat_in_wat_test.json')
    driver = vlx_interface.EmbeddingIntegralDriver(h2o_xyz, "cc-pvdz")
    # calculate nuclear es energy and electric fock matrix
    e_nuc_es, f_el_es = electrostatic_interactions.compute_electrostatic_interaction(quantum_subsystem=core,
                                                                                     classical_subsystem=env,
                                                                                     integral_drv=driver)
    E_s, C_s = vlx_interface.scf_solver(h=h + f_el_es, V_nuc=V_nuc + e_nuc_es, C=C_HF, nocc=nocc, g=g, S=S,
                                        conv_thresh=1e-4)
    D = 2 * np.einsum("ik,jk->ij", C_s[:, :nocc], C_s[:, :nocc])
    e_el_es = np.einsum("ab, ab", D, f_el_es)
    assert E_s == pytest.approx(-76.05504275, rel=1e-10)
    assert e_nuc_es == pytest.approx(-0.08545956, abs=1.5e-8)
    assert e_el_es == pytest.approx(0.00987780, abs=1.5e-8)
