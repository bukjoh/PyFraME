"""Tests PyFraME.embedding.subsystem.py"""
import os

import pytest
import veloxchem as vlx
import numpy as np
import scipy
from pyframe.embedding import subsystem, vlx_interface, induction_interactions, electrostatic_interactions

env = subsystem.ClassicalSubsystem(name="4x H2O atoms",
                                   input_data=f'{os.path.dirname(__file__)}/data/wat_in_wat_test.json')
core = subsystem.QuantumSubsystem(name="1x H2O",
                                  input_data=f'{os.path.dirname(__file__)}/data/wat_in_wat_test.json')
h_minus = subsystem.ClassicalSubsystem(name="H^{-}",
                                       input_data=f'{os.path.dirname(__file__)}/data/two_atom_test.json')
h2 = subsystem.QuantumSubsystem(name="H2",
                                input_data=f'{os.path.dirname(__file__)}/data/two_atom_test.json')


def test_compute_induction_interation():
    # Test H^{-} -(2AA) ENV|QM H -(0.74AA) H QM|ENV -(2AA) H^{-}
    # static contributions
    # VLX wants angstrom!
    h2_xyz = """2
        core H2                
        H        0.0000000000     0.0000000000      0.0000000000
        H        0.7399998775     0.0000000000      0.0000000000
        """
    basis = "sto-3g"
    driver = vlx_interface.EmbeddingIntegralDriver(h2_xyz, basis)
    static_drv = induction_interactions.compute_static_contributions
    coordinates, multipole_fields, nuclear_fields, polarizabilities, classical_fragments = (
        static_drv(quantum_subsystem=h2,
                   classical_subsystem=h_minus))
    static_fields = multipole_fields + nuclear_fields
    # induced dipoles
    ind_dipoles, electric_fields = induction_interactions.compute_induced_dipoles(density=h2.density_matrix.density,
                                                                                  integral_drv=driver,
                                                                                  coordinates=coordinates,
                                                                                  static_fields=static_fields,
                                                                                  polarizabilities=polarizabilities,
                                                                                  classical_fragments=classical_fragments,
                                                                                  threshold=1e-20)
    total_fields = static_fields + electric_fields
    ref_ind_dipole_1 = np.array([-0.039037184, 0., 0.])
    ref_ind_dipole_2 = np.array([0.039037184, 0., 0.])
    assert ind_dipoles[0, 0] == pytest.approx(ref_ind_dipole_1[0], abs=1e-8)
    assert ind_dipoles[1, 0] == pytest.approx(ref_ind_dipole_2[0], abs=1e-8)
    # induction energy and fock matrix contributions
    induction_energy, fock_matrix_contr = (induction_interactions.
                                           compute_induction_interaction(induced_dipoles=ind_dipoles,
                                                                         total_fields=total_fields,
                                                                         coordinates=coordinates,
                                                                         integral_drv=driver))
    # h2o test ind dipoles
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
    core_ac = subsystem.QuantumSubsystem(name="QM", input_data=f'{os.path.dirname(__file__)}/data/wat_in_wat_test.json')
    env_ac = subsystem.ClassicalSubsystem(name="Classical",
                                          input_data=f'{os.path.dirname(__file__)}/data/wat_in_wat_test.json')
    driver = vlx_interface.EmbeddingIntegralDriver(h2o_xyz, "cc-pvdz")
    # calculate nuclear es energy and electric fock matrix
    e_nuc_es, f_el_es = electrostatic_interactions.compute_electrostatic_interaction(quantum_subsystem=core_ac,
                                                                                     classical_subsystem=env_ac,
                                                                                     integral_drv=driver)
    E_s, C_s, ind_dip, e_ind, e_nuc_ind, e_mul_ind, e_el_ind = vlx_interface.scf_solver_with_ind(h=h + f_el_es,
                                                                                                 V_nuc=V_nuc + e_nuc_es,
                                                                                                 C=C_HF,
                                                                                                 nocc=nocc, g=g, S=S,
                                                                                                 embedding_driver=driver,
                                                                                                 core=core_ac,
                                                                                                 env=env_ac)
    D = 2 * np.einsum("ik,jk->ij", C_s[:, :nocc], C_s[:, :nocc])
    e_el_es = np.einsum("ab, ab", D, f_el_es)

    # setup test to compare to dalton

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
    core_ac = subsystem.QuantumSubsystem(name="1x C3OH4",
                                         input_data=f'{os.path.dirname(__file__)}/data/acrolein_test.json')
    env_ac = subsystem.ClassicalSubsystem(name="2x H2O atoms + X on their bonds",
                                          input_data=f'{os.path.dirname(__file__)}/data/acrolein_test.json')
    driver = vlx_interface.EmbeddingIntegralDriver(acrolein_xyz, basis_str)
    # calculate nuclear es energy and electric fock matrix
    e_nuc_es, f_el_es = electrostatic_interactions.compute_electrostatic_interaction(quantum_subsystem=core_ac,
                                                                                     classical_subsystem=env_ac,
                                                                                     integral_drv=driver)
    E_s, C_s, ind_dip, e_ind, e_nuc_ind, e_mul_ind, e_el_ind = vlx_interface.scf_solver_with_ind(h=h + f_el_es,
                                                                                                 V_nuc=V_nuc + e_nuc_es,
                                                                                                 C=C_HF,
                                                                                                 nocc=nocc, g=g, S=S,
                                                                                                 embedding_driver=driver,
                                                                                                 core=core_ac,
                                                                                                 env=env_ac)
    D = 2 * np.einsum("ik,jk->ij", C_s[:, :nocc], C_s[:, :nocc])
    e_el_es = np.einsum("ab, ab", D, f_el_es)
    e_pe_total = e_el_ind + e_el_es + e_mul_ind + e_nuc_ind + e_nuc_es
    assert e_el_es == pytest.approx(-0.281816731033, abs=1e-7)
    assert e_nuc_es == pytest.approx(0.266706671106, abs=1e-8)
    assert e_ind == pytest.approx(-0.000751933804, abs=1e-8)
    assert e_nuc_ind == pytest.approx(0.042124827087, abs=1.5e-7)
    assert e_mul_ind == pytest.approx(0.000114734523, abs=1e-9)
    assert e_el_ind == pytest.approx(-0.042991495415, abs=1.5e-7)
    assert e_pe_total == pytest.approx(-0.015861993732, abs=1e-7)
    assert E_s == pytest.approx(-188.314434428374, abs=1e-7)
    print(ind_dip)
    print(E_HF, E_s, e_mul_ind, e_nuc_ind, e_el_ind, e_ind, e_el_es, e_nuc_es, e_pe_total)