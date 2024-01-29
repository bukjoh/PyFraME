"""Tests PyFraME.embedding.subsystem.py"""
import os
import pytest
import veloxchem as vlx
import numpy as np
import scipy
from pyframe.embedding import subsystem, vlx_interface, induction_interactions, electrostatic_interactions, read_input


def test_compute_induction_interation():
    # Test H^{-} -(2AA) ENV|QM H -(0.74AA) H QM|ENV -(2AA) H^{-}
    # static contributions
    h2_xyz = """2
        core H2                
        H        0.0000000000     0.0000000000      0.0000000000
        H        0.7399998775     0.0000000000      0.0000000000
        """
    basis = "sto-3g"
    driver = vlx_interface.EmbeddingIntegralDriver(h2_xyz, basis)
    h2, h_minus = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/two_atom_test.json')
    electric_fields = h2.compute_electric_fields(coordinates=h_minus.coordinates, integral_drv=driver)
    nuclear_fields = h2.compute_nuclear_fields(coordinates=h_minus.coordinates)
    external_fields = electric_fields + nuclear_fields
    h_minus.solve_induced_dipoles(external_fields=external_fields, threshold=1e-10)
    total_fields =  h_minus.inducing_fields
    ref_ind_dipole_1 = np.array([-0.039037184, 0., 0.])
    ref_ind_dipole_2 = np.array([0.039037184, 0., 0.])
    ref_total_fields = np.array([])
    assert h_minus.induced_dipoles[0, 0] == pytest.approx(ref_ind_dipole_1[0], abs=1e-8)
    assert h_minus.induced_dipoles[1, 0] == pytest.approx(ref_ind_dipole_2[0], abs=1e-8)

    # echem test for induced dipoles
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
    # define core and env
    #env = subsystem.ClassicalSubsystem(name="4x H2O atoms",
    #                                   input_data=f'{os.path.dirname(__file__)}/data/wat_in_wat_test.json')
    #core = subsystem.QuantumSubsystem(name="1x H2O",
    #                                  input_data=f'{os.path.dirname(__file__)}/data/wat_in_wat_test.json')

    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/wat_in_wat_test.json')
    driver = vlx_interface.EmbeddingIntegralDriver(h2o_xyz, "cc-pvdz")
    # calculate nuclear es energy and electric fock matrix
    e_nuc_es, f_el_es = electrostatic_interactions.compute_electrostatic_interaction(quantum_subsystem=core,
                                                                                     classical_subsystem=env,
                                                                                     integral_drv=driver)
    E_s, C_s, ind_dip, e_ind, e_nuc_ind, e_mul_ind, e_el_ind = vlx_interface.scf_pe_solver(h=h + f_el_es,
                                                                                           V_nuc=V_nuc + e_nuc_es,
                                                                                           C=C_HF,
                                                                                           nocc=nocc, g=g, S=S,
                                                                                           embedding_driver=driver,
                                                                                           core=core,
                                                                                           env=env)
    D = 2 * np.einsum("ik,jk->ij", C_s[:, :nocc], C_s[:, :nocc])
    e_el_es = np.einsum("ab, ab", D, f_el_es)
    e_pe_total = e_el_ind + e_el_es + e_mul_ind + e_nuc_ind + e_nuc_es
    assert e_el_es == pytest.approx(0.006804775749414373, abs=1e-9)
    assert e_nuc_es == pytest.approx(-0.08545956246530774, abs=1e-9)
    assert e_ind == pytest.approx(-0.01006509940518837, abs=1e-9)
    assert e_nuc_ind == pytest.approx(-0.006052391164318287, abs=1e-9)
    assert e_mul_ind == pytest.approx(-0.0009277338262916619, abs=1e-9)
    assert e_el_ind == pytest.approx(-0.003084974414578435, abs=1e-9)
    assert e_pe_total == pytest.approx(-0.08871988612108175, abs=1e-9)
    assert E_s == pytest.approx(-76.06472693147695, rel=1e-10)
    # Test in comparison to dalton
    # define molecule
    acrolein_xyz = """8
    C3OH4
    C             -0.145335   -0.546770    0.000607
    C              1.274009   -0.912471   -0.000167
    C              1.630116   -2.207690   -0.000132
    O             -0.560104    0.608977    0.000534
    H             -0.871904   -1.386459    0.001253
    H              2.004448   -0.101417   -0.000710
    H              0.879028   -3.000685    0.000484
    H              2.675323   -2.516779   -0.000673
    """

    # vlx_interface.xyz_bohr_to_angstrom(acrolein_xyz)
    # return

    molecule = vlx.Molecule.read_xyz_string(acrolein_xyz)
    basis_str = "sto-3g"
    basis = vlx.MolecularBasis.read(molecule, basis_str)

    # better initial guess:
    scf_drv = vlx.ScfRestrictedDriver()
    scf_results = scf_drv.compute(molecule, basis)

    # scf preparation
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
    # initial guess -> does not converge here.
    epsilon, C = scipy.linalg.eigh(h, S)
    # therefore better guess from scf used here
    C = scf_results['C']
    E_HF, C_HF = vlx_interface.scf_solver(h=h, V_nuc=V_nuc, C=C, nocc=nocc, g=g, S=S)
    # define core and env
    #core_ac = subsystem.QuantumSubsystem(name="1x C3OH4",
    #                                     input_data=f'{os.path.dirname(__file__)}/data/acrolein_test.json')
    #env_ac = subsystem.ClassicalSubsystem(name="2x H2O atoms + X on their bonds",
    #                                      input_data=f'{os.path.dirname(__file__)}/data/acrolein_test.json')

    core_ac, env_ac = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/acrolein_test.json')
    driver = vlx_interface.EmbeddingIntegralDriver(acrolein_xyz, basis_str)
    # calculate nuclear es energy and electric fock matrix
    e_nuc_es, f_el_es = electrostatic_interactions.compute_electrostatic_interaction(quantum_subsystem=core_ac,
                                                                                     classical_subsystem=env_ac,
                                                                                     integral_drv=driver)
    E_s, C_s, ind_dip, e_ind, e_nuc_ind, e_mul_ind, e_el_ind = vlx_interface.scf_pe_solver(h=h + f_el_es,
                                                                                           V_nuc=V_nuc + e_nuc_es,
                                                                                           C=C_HF,
                                                                                           nocc=nocc, g=g, S=S,
                                                                                           embedding_driver=driver,
                                                                                           core=core_ac,
                                                                                           env=env_ac)
    D = 2 * np.einsum("ik,jk->ij", C_s[:, :nocc], C_s[:, :nocc])
    e_el_es = np.einsum("ab, ab", D, f_el_es)
    e_pe_total = e_el_ind + e_el_es + e_mul_ind + e_nuc_ind + e_nuc_es
    assert e_el_es == pytest.approx(-0.281816699842, abs=1e-8)
    assert e_nuc_es == pytest.approx(0.266706671106, abs=1e-8)
    assert e_ind == pytest.approx(-0.000751929428, abs=1e-9)
    assert e_nuc_ind == pytest.approx(0.042124675086, abs=1e-8)
    assert e_mul_ind == pytest.approx(0.000114734186, abs=1e-10)
    assert e_el_ind == pytest.approx(-0.0429913387, abs=1e-8)
    assert e_pe_total == pytest.approx(-0.015861958165, abs=1e-8)
    assert E_s == pytest.approx(-188.314434428389, rel=1e-10)

