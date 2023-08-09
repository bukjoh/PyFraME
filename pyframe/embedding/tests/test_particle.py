"""Tests PyFraME.embedding.particle.py"""
import pytest
import qcelemental
import copy
import numpy as np
from pyframe.embedding import particle, tensor_tools
from pyframe.embedding.tests import test_electrostatic_interactions
from qcelemental import PhysicalConstantsContext

phys_constants = PhysicalConstantsContext('CODATA2018')


def test_particle_init():
    particle.Particle(index=0,
                      mass=qcelemental.periodictable.to_mass('Ti'),
                      coordinate=(np.array([-5.3285510, -0.1032300, -0.0004160]) / phys_constants.bohr2angstroms))


def test_init_nucleus():
    particle.Nucleus(index=0, charge=8.0, element='O',
                     mass=qcelemental.periodictable.to_mass('O'),
                     coordinate=(np.array([-3.3285510, -0.1032300, -0.0004160])
                                 / phys_constants.bohr2angstroms))
    # Test element to nucleus
    with pytest.raises(ValueError):
        particle.Nucleus(index=0, charge=8.0, element='N',
                         mass=qcelemental.periodictable.to_mass('O'),
                         coordinate=(np.array([-3.3285510, -0.1032300, -0.0004160])
                                     / phys_constants.bohr2angstroms))


def test_init_atom():
    particle.Atom(index=0, element='O',
                  mass=qcelemental.periodictable.to_mass('O'),
                  coordinate=(np.array([1.7422970, 2.3413610, -0.0007450])
                              / phys_constants.bohr2angstroms),
                  multipoles={"elements": [-0.7424407021, -0.2840371815, 0.1671869101,
                                           0.0013661730, -4.4188113058, 0.2801417448,
                                           0.0002126277, -4.1129942280, 0.0039064597,
                                           -5.0206529493, 0.0330216712, 0.0760732530,
                                           -0.0003899377, 0.0087745562, 0.0009840165,
                                           -0.1813726316, -0.1218544851, -0.0016261873,
                                           0.1033223765, 0.0025668497],
                              "order": 3},
                  polarizabilities={"elements": np.array([0., 0., 0., 0., 2.2823444229, -0.4207398269,
                                                          -0.0006457588, 1.8324399300, -0.0069062619,
                                                          3.3666855109]), "order": [1, 1]})
    particle.Atom(index=1, element='H',
                  mass=qcelemental.periodictable.to_mass('H'),
                  coordinate=(np.array([0.8416780, 1.9718070, -0.0008200])
                              / phys_constants.bohr2angstroms),
                  multipoles={"elements": [0.3699635356, 0.1039146227, 0.0490621032,
                                           0.0000401115, -0.6455428297, -0.0041180935,
                                           0.0001453595, -0.5362599824, 0.0001224056,
                                           -0.5596286658, 0.0810557396, 0.0035896134,
                                           0.0001532545, -0.0090165544, -0.0002204117,
                                           0.0757377031, 0.2842212967, 0.0003840935,
                                           0.1046774555, 0.0011004803],
                              "order": 3},
                  polarizabilities={"elements": np.array([0., 0., 0., 0., 0.8132771942, 0.1387943320,
                                                          0.0004103386, 0.5814638105, -0.0000004496,
                                                          0.5927954021]), "order": [1, 1]})
    # Test detrace of Atom multipoles
    oxygen_atom = particle.Atom(index=0, element='O',
                                mass=qcelemental.periodictable.to_mass('O'),
                                coordinate=(np.array([-3.3285510, -0.1032300, -0.0004160])
                                            / phys_constants.bohr2angstroms),
                                multipoles={"elements": [0., 0., 0., 0., -3.9516312016, -0.0561791973,
                                                         0.0008348984, -4.5778807726, 0.0000430036,
                                                         -5.0206878337],
                                            "order": 2})

    # Test detrace of multipoles in atom
    quadrupole = np.array([-3.9516312016, -0.0561791973, 0.0008348984, -4.5778807726, 0.0000430036, -5.0206878337])
    traceless_quadrupole = copy.deepcopy(quadrupole)
    trace_quadrupole = np.sum([quadrupole[0], quadrupole[3], quadrupole[5]]) / 3.0
    traceless_quadrupole[[0, 3, 5]] -= trace_quadrupole
    assert np.allclose(oxygen_atom.multipoles.data[4:10], traceless_quadrupole)
    # Test taylor_coefficient
    ref_taylor_coefficient = np.array([1., -1., -1., -1., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    assert np.allclose(oxygen_atom.taylor_coefficients.data, ref_taylor_coefficient)
    # Test degeneracy
    ref_degeneracy_tensor = np.array([1., 1., 1., 1., 1., 2., 2., 1., 2., 1.])
    assert np.allclose(ref_degeneracy_tensor, oxygen_atom.degeneracy_tensor.data)
    # Test multipole_with_degeneracy
    ref_degeneracy_tensor = np.array([1., 1., 1., 1., 1., 2., 2., 1., 2., 1.])
    ref_multipole_tensor = np.array([0., 0., 0., 0., -3.9516312016, -0.0561791973, 0.0008348984, -4.5778807726,
                                     0.0000430036, -5.0206878337])
    ref_multipole_tensor[4:10] = tensor_tools.detrace(ref_multipole_tensor[4:10],
                                                      oxygen_atom.particle_variables.factorials,
                                                      oxygen_atom.particle_variables.double_factorials,
                                                      oxygen_atom.particle_variables.trinomials)
    ref_multipole_tensor_with_degeneracy = np.multiply(ref_multipole_tensor, ref_degeneracy_tensor)
    assert np.allclose(ref_multipole_tensor_with_degeneracy, oxygen_atom.multipoles_with_degeneracy.data)


def test_potential():
    # For Atom.potential() and Nucleus.potential
    # Test 0th order derivative
    oxygen_atom = test_electrostatic_interactions.oxygen_atom
    hydrogen_atom = test_electrostatic_interactions.hydrogen_atom
    oxygen_nucleus = test_electrostatic_interactions.oxygen_nucleus
    hydrogen1_nucleus = test_electrostatic_interactions.hydrogen1_nucleus
    hydrogen2_nucleus = test_electrostatic_interactions.hydrogen2_nucleus
    assert hydrogen_atom.potential(coordinate=oxygen_atom.coordinate) \
           == pytest.approx(0.2023997479735371, 1e-9)
    assert oxygen_atom.potential(coordinate=oxygen_nucleus.coordinate) \
           == pytest.approx(-6.7735786708826934e-2, 1e-9)
    assert oxygen_nucleus.potential(coordinate=oxygen_atom.coordinate) \
           == pytest.approx(0.75202669489040475, 1e-9)
    assert hydrogen1_nucleus.potential(coordinate=hydrogen2_nucleus.coordinate,
                                       pot_derivative_order=0,
                                       origin_derivative_order=0) \
           == pytest.approx(0.34335113566514336, 1e-9)
    assert hydrogen2_nucleus.potential(coordinate=hydrogen1_nucleus.coordinate,
                                       pot_derivative_order=0,
                                       origin_derivative_order=0) \
           == pytest.approx(0.34335113566514336, 1e-9)
    assert hydrogen1_nucleus.potential(coordinate=hydrogen1_nucleus.coordinate,
                                       pot_derivative_order=0,
                                       origin_derivative_order=0) \
           == float('inf')
    # Test 1st order derivative
    ref = np.array([-6.3679443173877762e-2, -3.0699045537920477e-2, 4.1315647410858666e-6])
    assert np.allclose(oxygen_nucleus.potential(coordinate=oxygen_atom.coordinate,
                                                pot_derivative_order=1), ref)
    ref = np.array([-5.5956485010021746e-3, -2.5226526034117761e-3, 1.0334216193982794e-6])
    assert np.allclose(oxygen_atom.potential(coordinate=oxygen_nucleus.coordinate,
                                             origin_derivative_order=1), ref * (-1))
    assert np.allclose(oxygen_atom.potential(coordinate=oxygen_nucleus.coordinate,
                                             pot_derivative_order=1,
                                             origin_derivative_order=0), ref)
    ref = np.array([1.17446815e-01, -1.02122543e-02, 9.09486352e-05])
    assert np.allclose(hydrogen1_nucleus.potential(coordinate=hydrogen2_nucleus.
                                                   coordinate,
                                                   pot_derivative_order=0,
                                                   origin_derivative_order=1), ref * (-1))
    assert np.allclose(hydrogen1_nucleus.potential(coordinate=hydrogen2_nucleus.
                                                   coordinate,
                                                   pot_derivative_order=1,
                                                   origin_derivative_order=0), ref)
    assert np.allclose(hydrogen2_nucleus.potential(coordinate=hydrogen1_nucleus.
                                                   coordinate,
                                                   pot_derivative_order=0,
                                                   origin_derivative_order=1), ref)
    assert np.allclose(hydrogen2_nucleus.potential(coordinate=hydrogen1_nucleus.
                                                   coordinate,
                                                   pot_derivative_order=1,
                                                   origin_derivative_order=0), ref * (-1))
    for i in range(3):
        assert np.isnan(hydrogen1_nucleus.potential(coordinate=hydrogen1_nucleus.coordinate,
                                                    pot_derivative_order=0,
                                                    origin_derivative_order=1)[i])
    # Test 2nd order derivative
    ref = np.array([-8.1960563414034619e-4, -6.2660333024683009e-4, 1.8883791746125624e-7,
                    2.7461891397905321e-4, 1.7140147595889211e-7, 5.4498672016129260e-4])
    assert np.allclose(oxygen_atom.potential(coordinate=oxygen_nucleus.coordinate,
                                             origin_derivative_order=2), ref)
    assert np.allclose(oxygen_atom.potential(coordinate=oxygen_nucleus.coordinate,
                                             pot_derivative_order=1,
                                             origin_derivative_order=1), ref * (-1))
    # Test 3rd order derivative
    ref = np.array([-1.5061439556848243e-4, -1.9495759751591218e-4, 2.9483606775445033e-8,
                    1.9229830800340373e-5, 5.8440852920764875e-8, 1.3138456476814194e-4,
                    1.3470095565551845e-4, 2.3611770228146266e-8, 6.0256641860393659e-5,
                    -5.3095377003591325e-8])
    assert np.allclose(oxygen_atom.potential(coordinate=oxygen_nucleus.coordinate,
                                             pot_derivative_order=0,
                                             origin_derivative_order=3), ref * (-1))
    assert np.allclose(oxygen_atom.potential(coordinate=oxygen_nucleus.coordinate,
                                             pot_derivative_order=3,
                                             origin_derivative_order=0), ref)
    assert np.allclose(oxygen_atom.potential(coordinate=oxygen_nucleus.coordinate,
                                             pot_derivative_order=2,
                                             origin_derivative_order=1), ref * (-1))


def test_init_virtual_particle():
    particle.VirtualParticle(index=0, coordinate=np.array([1.2919875, 2.156584, -0.0007825])
                                                 / phys_constants.bohr2angstroms)
