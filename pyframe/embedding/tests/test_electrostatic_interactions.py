"""Tests PyFraME.embedding.electrostatic_interactions.py"""
import json
import os
import qcelemental
import pytest
import numpy as np
import scipy
import veloxchem as vlx
from pyframe.embedding import (particle, polytensor, electrostatic_interactions, constants, fragment, subsystem,
                               vlx_interface)
from qcelemental import PhysicalConstantsContext

phys_constants = PhysicalConstantsContext('CODATA2018')

# Setup Oxygen Atom and Oxygen Nucleus
oxygen_nucleus = particle.Nucleus(index=0, charge=8.0, element='O',
                                  mass=qcelemental.periodictable.to_mass('O'),
                                  coordinate=(np.array([-3.3285510, -0.1032300, -0.0004160])
                                              / phys_constants.bohr2angstroms))
oxygen_atom = particle.Atom(index=0, element='O',
                            mass=qcelemental.periodictable.to_mass('O'),
                            coordinate=(np.array([1.7422970, 2.3413610, -0.0007450]) / phys_constants.bohr2angstroms),
                            multipoles={'elements': np.array([-0.7424407021, -0.2840371815, 0.1671869101,
                                                              0.0013661730, -4.4188113058, 0.2801417448,
                                                              0.0002126277, -4.1129942280, 0.0039064597,
                                                              -5.0206529493, 0.0330216712, 0.0760732530,
                                                              -0.0003899377, 0.0087745562, 0.0009840165,
                                                              -0.1813726316, -0.1218544851, -0.0016261873,
                                                              0.1033223765, 0.0025668497])},
                            polarizabilities={'elements': np.array([0., 0., 0., 0., 2.2823444229, -0.4207398269,
                                                                    -0.0006457588, 1.8324399300, -0.0069062619,
                                                                    3.3666855109]), 'order': [1, 1]})
# Setup Hydrogen Nuclei
hydrogen1_nucleus = particle.Nucleus(index=1, charge=1.0, element='H', mass=qcelemental.periodictable.to_mass('H'),
                                     coordinate=(np.array([-2.5037950, 0.4132210, 0.0003390])
                                                 / phys_constants.bohr2angstroms))
hydrogen2_nucleus = particle.Nucleus(index=2, charge=1.0, element='H', mass=qcelemental.periodictable.to_mass('H'),
                                     coordinate=(np.array([-4.0392140, 0.5467290, -0.0008500])
                                                 / phys_constants.bohr2angstroms))
# Setup Hydrogen Atom
hydrogen_atom = particle.Atom(index=1, element='H',
                              mass=qcelemental.periodictable.to_mass('H'),
                              coordinate=(np.array([0.8416780, 1.9718070, -0.0008200])
                                          / phys_constants.bohr2angstroms),
                              multipoles={'elements': np.array([0.3699635356, 0.1039146227, 0.0490621032,
                                                                0.0000401115, -0.6455428297, -0.0041180935,
                                                                0.0001453595, -0.5362599824, 0.0001224056,
                                                                -0.5596286658, 0.0810557396, 0.0035896134,
                                                                0.0001532545, -0.0090165544, -0.0002204117,
                                                                0.0757377031, 0.2842212967, 0.0003840935,
                                                                0.1046774555, 0.0011004803])},
                              polarizabilities={'elements': np.array([0., 0., 0., 0., 0.8132771942, 0.1387943320,
                                                                      0.0004103386, 0.5814638105, -0.0000004496,
                                                                      0.5927954021]), 'order': [1, 1]})
# Setup Hydrogen Atoms for Monopole Electrostatic energy test
hydrogen_atom1 = particle.Atom(index=1, element='H',
                               mass=qcelemental.periodictable.to_mass('H'),
                               coordinate=(np.array([1.0, 0.0, 0.0])
                                           / phys_constants.bohr2angstroms),
                               multipoles={'elements': np.array([1.0])}, )
hydrogen_atom2 = particle.Atom(index=1, element='H',
                               mass=qcelemental.periodictable.to_mass('H'),
                               coordinate=(np.array([1.0, 1.0, 0.0])
                                           / phys_constants.bohr2angstroms),
                               multipoles={'elements': np.array([1.0])})
# Setup Hydrogen Atoms for Dipole Electrostatic energy test
hydrogen_atom_dipole_z = particle.Atom(index=1, element='H',
                                       mass=qcelemental.periodictable.to_mass('H'),
                                       coordinate=(np.array([0.0, 0.0, 0.0])
                                                   / phys_constants.bohr2angstroms),
                                       multipoles={'elements': np.array([0.0, 0.0, 0.0, 1.0])})
hydrogen_atom_dipole_y = particle.Atom(index=1, element='H',
                                       mass=qcelemental.periodictable.to_mass('H'),
                                       coordinate=(np.array([1.0, 0.0, 0.0])
                                                   / phys_constants.bohr2angstroms),
                                       multipoles={'elements': np.array([1.0, 0.0, 1.0, 0.0])})
hydrogen_atom_dipole_z_translated_x = particle.Atom(index=1, element='H',
                                                    mass=qcelemental.periodictable.to_mass('H'),
                                                    coordinate=(np.array([1.0, 0.0, 0.0])
                                                                / phys_constants.bohr2angstroms),
                                                    multipoles={'elements': np.array([0.0, 0.0, 0.0, 1.0])})
hydrogen_atom_dipole_z_translated_z = particle.Atom(index=1, element='H',
                                                    mass=qcelemental.periodictable.to_mass('H'),
                                                    coordinate=(np.array([0.0, 0.0, 1.0])
                                                                / phys_constants.bohr2angstroms),
                                                    multipoles={'elements': np.array([0.0, 0.0, 0.0, 1.0])})
hydrogen_atom_dipole_translated = particle.Atom(index=1, element='H',
                                                mass=qcelemental.periodictable.to_mass('H'),
                                                coordinate=(np.array([5.0, 13.0, 1.0])
                                                            / phys_constants.bohr2angstroms),
                                                multipoles={'elements': np.array([0.0, 2.0, -7.0, 10.0])})
# Setup fragment dictionary
water_fragment_dict = {
    "index": 1,
    "name": "H2O",
    "atoms": [{
        "index": 1,
        "element": "O",
        "coordinate": [29.514, 37.243, 44.334],
        "multipoles": {'elements': [-0.71543374, 0.11412407, -0.27166543, 0.07772714, -4.71453229, -0.05566867,
                                    0.46147879, -4.19504704, 0.33577098, -3.77169662]},
        "polarizabilities": {'elements': [4.70788802, 0.33755124, -0.41867523, 3.74951294, -0.04025344, 4.09400356],
                             'order': [1, 1]}
    }, {
        "index": 2,
        "element": "H",
        "coordinate": [29.502, 36.434, 43.822],
        "multipoles": {'elements': [0.35771989, 0.00677109, 0.14724014, 0.1028778, -0.44388795, -0.00418109, 0.0053025,
                                    -0.13726601, 0.17783555, -0.32940875]},
        "polarizabilities": {'elements': [0.54070423, 0.05950662, -0.1316576, 2.1100446, 1.02222807, 0.95665643],
                             'order': [1, 1]}
    }, {
        "index": 3,
        "element": "H",
        "coordinate": [29.965, 37.007, 45.145],
        "multipoles": {'elements': [0.35771384, -0.08352935, 0.03548195, -0.15515376, -0.35386318, -0.05563024,
                                    0.16460186, -0.40252281, -0.09043114, -0.1541939]},
        "polarizabilities": {'elements': [1.08814028, -0.25332613, 0.83702697, 0.49714845, -0.60912036, 2.02223557],
                             'order': [1, 1]}
    }]
}

# Read fragments from JSON
with open(f'{os.path.dirname(__file__)}/data/act_wat_test.json') as json_file:
    input_data = json.load(json_file).get('classical_subsystem', None)

classical_fragments = []
for f in input_data['classical_fragments']:
    classical_fragments.append(fragment.ClassicalFragment(**f))


def test_compute_atoms_interaction():
    # Test Monopole-Monopole interaction
    ref_interaction_energy_monopole = 2.30707755234174E-18
    assert electrostatic_interactions.compute_atoms_interaction(hydrogen_atom1, hydrogen_atom2) \
           * phys_constants.hartree2J == pytest.approx(ref_interaction_energy_monopole, 1e-9)
    assert electrostatic_interactions.compute_atoms_interaction(hydrogen_atom2, hydrogen_atom1) \
           * phys_constants.hartree2J == pytest.approx(ref_interaction_energy_monopole, 1e-9)
    # Test orthogonal Dipole-Dipole interaction
    ref_interaction_energy_orth_dipole = 0.0
    assert electrostatic_interactions.compute_atoms_interaction(hydrogen_atom_dipole_z,
                                                                hydrogen_atom_dipole_y) \
           * phys_constants.hartree2J == pytest.approx(ref_interaction_energy_orth_dipole, 1e-9)
    # Test parallel Dipole-Dipole interaction
    ref_interaction_energy_parallel_dipole_xz = 6.46047513751174E-19
    assert electrostatic_interactions.compute_atoms_interaction(hydrogen_atom_dipole_z,
                                                                hydrogen_atom_dipole_z_translated_x) \
           * phys_constants.hartree2J == \
           pytest.approx(ref_interaction_energy_parallel_dipole_xz, 1e-9)
    ref_interaction_energy_parallel_dipole_zz = -1.29209502750235E-18
    assert electrostatic_interactions.compute_atoms_interaction(hydrogen_atom_dipole_z,
                                                                hydrogen_atom_dipole_z_translated_z) \
           * phys_constants.hartree2J == \
           pytest.approx(ref_interaction_energy_parallel_dipole_zz, 1e-9)
    # Test multi-directional Dipole-Dipole interaction
    ref_interaction_energy_dipole = 2.63168830505711E-21
    assert electrostatic_interactions.compute_atoms_interaction(hydrogen_atom_dipole_z,
                                                                hydrogen_atom_dipole_translated) \
           * phys_constants.hartree2J == \
           pytest.approx(ref_interaction_energy_dipole, 1e-9)


def test_compute_nucleus_atom_interaction():
    # Test compute_nucleus_atom_interaction against compute_atom_nucleus_interaction

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
    taylor_coefficient = oxygen_atom.taylor_coefficients.data
    ref_interaction_energy = ref_polytensor.dot_first_degree(polytensor.FirstDegreePolytensor.
                                                             multiply_elementwise(oxygen_atom.
                                                                                  multipoles_with_degeneracy,
                                                                                  taylor_coefficient))
    assert electrostatic_interactions.compute_atom_nucleus_interaction(oxygen_atom, oxygen_nucleus) \
           == pytest.approx(ref_interaction_energy, 1e-9)


def test_compute_nuclei_interaction():
    assert electrostatic_interactions.compute_nuclei_interaction(hydrogen1_nucleus, hydrogen2_nucleus) \
           == pytest.approx(0.34335113566, 1e-9)
    assert electrostatic_interactions.compute_nuclei_interaction(hydrogen1_nucleus, hydrogen2_nucleus) \
           == electrostatic_interactions.compute_nuclei_interaction(hydrogen2_nucleus, hydrogen1_nucleus)
    assert electrostatic_interactions.compute_nuclei_interaction(hydrogen1_nucleus, hydrogen1_nucleus) \
           == float('inf')
    assert electrostatic_interactions.compute_nuclei_interaction(oxygen_nucleus, hydrogen2_nucleus) \
           == pytest.approx(4.39578846868, 1e-9)
    assert electrostatic_interactions.compute_nuclei_interaction(oxygen_nucleus, hydrogen1_nucleus) \
           == pytest.approx(4.35039628273, 1e-9)


def test_compute_t_tensor():
    # Test with interaction_tensor_template
    ref_potential = np.array(
        [0.75202669489040475, 6.3679443173877762e-2, 3.0699045537920477e-2, -4.1315647410858666e-6,
         9.5311929756771267e-3, 7.7985188788112876e-3, -1.0495468203592808e-6,
         -2.8858133904975198e-3, -5.0597310570715463e-7, -6.6453795851796043e-3,
         1.7845230927165381e-3, 2.4879519222560046e-3, -3.3483563607254773e-7,
         -9.6388595705967816e-5, -2.1422168821470319e-7, -1.6881344970105696e-3,
         -1.6741238607204253e-3, 6.2537563711756675e-9, -8.1382806153557845e-4,
         3.2858187970137201e-7], dtype=np.float64) / 8.0
    t_tensor = electrostatic_interactions.compute_t_tensor(r_a=oxygen_nucleus.coordinate,
                                                           r_b=oxygen_atom.coordinate,
                                                           rank_a=0,
                                                           rank_b=oxygen_atom.multipole_order,
                                                           tensor_template=constants.values.
                                                           interaction_tensor_template).data

    assert np.allclose(t_tensor, ref_potential)
    # Test with potential_tensor_template
    ref_potential = np.array(
        [0.75202669489040475, -6.3679443173877762e-2, -3.0699045537920477e-2, 4.1315647410858666e-6,
         9.5311929756771267e-3, 7.7985188788112876e-3, -1.0495468203592808e-6,
         -2.8858133904975198e-3, -5.0597310570715463e-7, -6.6453795851796043e-3,
         -1.7845230927165381e-3, -2.4879519222560046e-3, 3.3483563607254773e-7,
         9.6388595705967816e-5, 2.1422168821470319e-7, 1.6881344970105696e-3,
         1.6741238607204253e-3, -6.2537563711756675e-9, 8.1382806153557845e-4,
         -3.2858187970137201e-7], dtype=np.float64) / 8.0
    t_tensor = electrostatic_interactions.compute_t_tensor(r_a=oxygen_nucleus.coordinate,
                                                           r_b=oxygen_atom.coordinate,
                                                           rank_a=0,
                                                           rank_b=oxygen_atom.multipole_order,
                                                           tensor_template=constants.values.
                                                           potential_tensor_template).data

    assert np.allclose(t_tensor, ref_potential)


def test_compute_fragment_nucleus_interaction():
    water_fragment = fragment.ClassicalFragment(**water_fragment_dict)
    interaction_energy = electrostatic_interactions.compute_atom_nucleus_interaction
    es_energy = electrostatic_interactions.compute_fragment_nucleus_interaction(oxygen_nucleus,
                                                                                water_fragment)
    ref_energy = interaction_energy(water_fragment.atoms[0], oxygen_nucleus) \
                 + interaction_energy(water_fragment.atoms[1], oxygen_nucleus) \
                 + interaction_energy(water_fragment.atoms[2], oxygen_nucleus)
    assert ref_energy == pytest.approx(es_energy, 1e-9)


def test_compute_fragment_atom_interaction():
    water_fragment = fragment.ClassicalFragment(**water_fragment_dict)
    interaction_energy = electrostatic_interactions.compute_atoms_interaction
    es_energy = electrostatic_interactions.compute_fragment_atom_interaction(oxygen_atom,
                                                                             water_fragment)
    ref_energy = interaction_energy(water_fragment.atoms[0], oxygen_atom) \
                 + interaction_energy(water_fragment.atoms[1], oxygen_atom) \
                 + interaction_energy(water_fragment.atoms[2], oxygen_atom)
    assert ref_energy == pytest.approx(es_energy, 1e-9)


def test_fragments_interaction():
    water_fragment_1 = classical_fragments[0]
    water_fragment_2 = classical_fragments[1]
    interaction_energy = electrostatic_interactions.compute_atoms_interaction
    es_energy = electrostatic_interactions.compute_fragments_interaction(water_fragment_1, water_fragment_2)
    ref_energy = 0
    for atoms in water_fragment_2.atoms:
        ref_energy += interaction_energy(water_fragment_1.atoms[0], atoms) \
                      + interaction_energy(water_fragment_1.atoms[1], atoms) \
                      + interaction_energy(water_fragment_1.atoms[2], atoms)
    assert ref_energy == pytest.approx(es_energy, 1e-9)


def test_compute_electrostatic_interaction():
    # Internal test
    core = subsystem.QuantumSubsystem(name="QM", input_data=f'{os.path.dirname(__file__)}/data/act_wat_test.json')
    env = subsystem.ClassicalSubsystem(name="Classical",
                                       input_data=f'{os.path.dirname(__file__)}/data/act_wat_test.json')
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
            ref_energy += electrostatic_interactions.compute_fragment_nucleus_interaction(nucleus, fragments)
        for atom in env.atoms:
            ref_energy += electrostatic_interactions.compute_atom_nucleus_interaction(atom, nucleus)
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
    env = subsystem.ClassicalSubsystem(name="4x H2O atoms",
                                       input_data=f'{os.path.dirname(__file__)}/data/wat_in_wat_test.json')
    core = subsystem.QuantumSubsystem(name="1x H2O",
                                      input_data=f'{os.path.dirname(__file__)}/data/wat_in_wat_test.json')
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
