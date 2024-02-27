"""Tests PyFraME.embedding.particle.py"""
import pytest
import qcelemental
import copy
import numpy as np

from pyframe.embedding import particle, tensor_tools, constants, polytensor


class TestParticle:
    def test_init_with_valid_arguments(self,
                                       phys_constants
                                       ):
        index = 1
        coordinate = np.array([-5.3285510, -0.1032300, -0.0004160]) / phys_constants.bohr2angstroms
        mass = qcelemental.periodictable.to_mass('Ti')
        particle_inst = particle.Particle(index=index, coordinate=coordinate, mass=mass)
        assert particle_inst.index == index
        assert particle_inst.coordinate is coordinate
        assert particle_inst._mass == mass
        assert particle_inst.particle_variables is constants.values


class TestAtom:
    def test_init_with_valid_arguments(self, oxygen_data, hydrogen_data):
        index = 1
        coordinate = np.array([0.0, 0.0, 0.0])
        induced_dipole = np.array([1.0, 2.0, 3.0])
        name = "Atom1"
        exclusions = [2, 3]
        mass = 12.01
        element = "C"
        vdw = {'vdw_method': "6-12", 'lj_sigma': 3.0, 'lj_epsilon': 0.5}
        multipoles = {'elements': [-0.71543374,
                                   0.11412407,
                                   -0.27166543,
                                   0.07772714,
                                   -4.71453229,
                                   -0.05566867,
                                   0.46147879,
                                   -4.19504704,
                                   0.33577098,
                                   -3.77169662],
                      'order': 2}
        polarizabilities = {"elements": [0.0,
                                         0.0,
                                         0.0,
                                         0.0,
                                         4.70788802,
                                         0.33755124,
                                         -0.41867523,
                                         3.74951294,
                                         -0.04025344,
                                         4.09400356
                                         ],
                            "order": [1, 1]}
        atom = particle.Atom(index=index,
                             coordinate=coordinate,
                             induced_dipole=induced_dipole,
                             name=name,
                             exclusions=exclusions,
                             mass=mass,
                             element=element,
                             vdw=vdw,
                             multipoles=multipoles,
                             polarizabilities=polarizabilities)
        assert atom.index == index
        assert atom.coordinate is coordinate
        assert np.array_equal(atom.induced_dipole, induced_dipole)
        assert atom.name == name
        assert atom.exclusions == tuple(exclusions)
        assert atom._mass == mass
        assert atom._element == element
        assert atom._vdw_method == vdw['vdw_method']
        assert atom._lj_sigma == vdw['lj_sigma']
        assert atom._lj_epsilon == vdw['lj_epsilon']
        assert atom.multipole_order == multipoles['order']
        assert isinstance(atom.multipoles, polytensor.FirstDegreePolytensor)
        assert np.array_equal(atom.polarizability_order, polarizabilities['order'])
        assert np.array_equal(atom.polarizability, np.array(polarizabilities['elements']))
        # Create Atom instances
        oxygen_atom = particle.Atom(**oxygen_data)
        hydrogen_atom = particle.Atom(**hydrogen_data)

        # Test attributes
        assert oxygen_atom.index == oxygen_data["index"]
        assert np.array_equal(oxygen_atom.coordinate, oxygen_data["coordinate"])
        assert np.array_equal(oxygen_atom.induced_dipole, None)
        assert oxygen_atom.name is None
        assert np.array_equal(oxygen_atom.exclusions, oxygen_data["exclusions"])
        assert oxygen_atom._element == oxygen_data["element"]
        assert oxygen_atom.multipole_order == oxygen_data["multipoles"]["order"]
        assert isinstance(oxygen_atom.multipoles, polytensor.FirstDegreePolytensor)
        assert np.array_equal(oxygen_atom.polarizability_order, oxygen_data["polarizabilities"]["order"])
        assert np.array_equal(oxygen_atom.polarizability, oxygen_data["polarizabilities"]["elements"])

        assert hydrogen_atom.index == hydrogen_data["index"]
        assert np.array_equal(hydrogen_atom.coordinate, hydrogen_data["coordinate"])
        assert np.array_equal(hydrogen_atom.induced_dipole, None)
        assert hydrogen_atom.name is None
        assert np.array_equal(hydrogen_atom.exclusions, hydrogen_data["exclusions"])
        assert hydrogen_atom._element == hydrogen_data["element"]
        assert hydrogen_atom.multipole_order == hydrogen_data["multipoles"]["order"]
        assert isinstance(hydrogen_atom.multipoles, polytensor.FirstDegreePolytensor)
        assert np.array_equal(hydrogen_atom.polarizability_order, hydrogen_data["polarizabilities"]["order"])
        assert np.array_equal(hydrogen_atom.polarizability, hydrogen_data["polarizabilities"]["elements"])

    def test_init_with_invalid_exclusions(self):
        with pytest.raises(ValueError, match="Exclusions must be a list."):
            particle.Atom(index=1, coordinate=np.array([0.0, 0.0, 0.0]), exclusions="invalid_input")

    def test_detrace_of_atom_multipoles(self, oxygen_atom2):
        # Expected result after detracing
        quadrupole = np.array([-3.9516312016, -0.0561791973, 0.0008348984, -4.5778807726, 0.0000430036, -5.0206878337])
        traceless_quadrupole = copy.deepcopy(quadrupole)
        trace_quadrupole = np.sum([quadrupole[0], quadrupole[3], quadrupole[5]]) / 3.0
        traceless_quadrupole[[0, 3, 5]] -= trace_quadrupole

        # Test detrace
        assert np.allclose(oxygen_atom2.multipoles.data[4:10], traceless_quadrupole)

    def test_taylor_coefficient(self,
                                oxygen_atom2
                                ):
        # Expected result for taylor_coefficient
        ref_taylor_coefficient = np.array([1., -1., -1., -1., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        # Test taylor_coefficient
        assert np.allclose(oxygen_atom2.taylor_coefficients.data, ref_taylor_coefficient)

    def test_degeneracy(self,
                        oxygen_atom2
                        ):
        # Expected result for degeneracy tensor
        ref_degeneracy_tensor = np.array([1., 1., 1., 1., 1., 2., 2., 1., 2., 1.])

        # Test degeneracy tensor
        assert np.allclose(ref_degeneracy_tensor, oxygen_atom2.degeneracy_tensor.data)

    def test_multipole_with_degeneracy(self,
                                       oxygen_atom2
                                       ):
        # Expected result for multipole_with_degeneracy
        ref_degeneracy_tensor = np.array([1., 1., 1., 1., 1., 2., 2., 1., 2., 1.])
        ref_multipole_tensor = np.array([0., 0., 0., 0., -3.9516312016, -0.0561791973, 0.0008348984, -4.5778807726,
                                         0.0000430036, -5.0206878337])
        ref_multipole_tensor[4:10] = tensor_tools.detrace(ref_multipole_tensor[4:10],
                                                          oxygen_atom2.particle_variables.factorials,
                                                          oxygen_atom2.particle_variables.double_factorials,
                                                          oxygen_atom2.particle_variables.trinomials)
        ref_multipole_tensor_with_degeneracy = np.multiply(ref_multipole_tensor, ref_degeneracy_tensor)

        # Test multipole_with_degeneracy
        assert np.allclose(ref_multipole_tensor_with_degeneracy, oxygen_atom2.multipoles_with_degeneracy.data)

    def test_multipole_len_to_order(self):
        x = 10
        expected_order = 2
        result = particle.multipole_len_to_order(x)
        assert result == expected_order
        # Test case with a negative input, expecting a ValueError
        x = -5
        with pytest.raises(ValueError, match="Input must be a non-negative integer."):
            particle.multipole_len_to_order(x)

    def test_potential(self,
                       hydrogen_atom,
                       oxygen_atom1,
                       oxygen_nucleus,

                       ):
        # Test 0th order derivative
        assert hydrogen_atom.potential(coordinate=oxygen_atom1.coordinate) \
               == pytest.approx(0.2023997479735371, 1e-9)
        assert oxygen_atom1.potential(coordinate=oxygen_nucleus.coordinate) \
               == pytest.approx(-6.7735786708826934e-2, 1e-9)
        # Test 1st order derivative
        ref = np.array([-5.5956485010021746e-3, -2.5226526034117761e-3, 1.0334216193982794e-6])
        assert np.allclose(oxygen_atom1.potential(coordinate=oxygen_nucleus.coordinate,
                                                  origin_derivative_order=1), ref * (-1))
        assert np.allclose(oxygen_atom1.potential(coordinate=oxygen_nucleus.coordinate,
                                                  pot_derivative_order=1,
                                                  origin_derivative_order=0), ref)
        # Test 2nd order derivative
        ref = np.array([-8.1960563414034619e-4, -6.2660333024683009e-4, 1.8883791746125624e-7,
                        2.7461891397905321e-4, 1.7140147595889211e-7, 5.4498672016129260e-4])
        assert np.allclose(oxygen_atom1.potential(coordinate=oxygen_nucleus.coordinate,
                                                  origin_derivative_order=2), ref)
        assert np.allclose(oxygen_atom1.potential(coordinate=oxygen_nucleus.coordinate,
                                                  pot_derivative_order=1,
                                                  origin_derivative_order=1), ref * (-1))
        # Test 3rd order derivative
        ref = np.array([-1.5061439556848243e-4, -1.9495759751591218e-4, 2.9483606775445033e-8,
                        1.9229830800340373e-5, 5.8440852920764875e-8, 1.3138456476814194e-4,
                        1.3470095565551845e-4, 2.3611770228146266e-8, 6.0256641860393659e-5,
                        -5.3095377003591325e-8])
        assert np.allclose(oxygen_atom1.potential(coordinate=oxygen_nucleus.coordinate,
                                                  pot_derivative_order=0,
                                                  origin_derivative_order=3), ref * (-1))
        assert np.allclose(oxygen_atom1.potential(coordinate=oxygen_nucleus.coordinate,
                                                  pot_derivative_order=3,
                                                  origin_derivative_order=0), ref)
        assert np.allclose(oxygen_atom1.potential(coordinate=oxygen_nucleus.coordinate,
                                                  pot_derivative_order=2,
                                                  origin_derivative_order=1), ref * (-1))


class TestNucleus:
    def test_creation(self):
        index = 1
        coordinate = np.array([0.0, 0.0, 0.0])
        charge = 2.0
        nucleus = particle.Nucleus(index, coordinate, charge)
        assert nucleus.index == index
        assert np.array_equal(nucleus.coordinate, coordinate)
        assert nucleus.charge == np.array([charge])

    def test_charge_to_element(self,
                               phys_constants
                               ):
        index = 1
        coordinate = np.array([0.0, 0.0, 0.0])
        charge = 2.0
        nucleus = particle.Nucleus(index, coordinate, charge)
        assert nucleus.charge_to_element() == "Helium"  # Assuming atomic number 2 corresponds to Helium
        # Test with wrong input
        with pytest.raises(ValueError):
            particle.Nucleus(index=0, charge=8.0, element='N',
                             mass=qcelemental.periodictable.to_mass('O'),
                             coordinate=(np.array([-3.3285510, -0.1032300, -0.0004160])
                                         / phys_constants.bohr2angstroms))

    def test_element_to_charge(self):
        index = 1
        coordinate = np.array([0.0, 0.0, 0.0])
        charge = 2.0
        nucleus = particle.Nucleus(index, coordinate, charge)
        assert nucleus.element_to_charge() == charge

    def test_potential(self,
                       oxygen_nucleus,
                       hydrogen1_nucleus,
                       hydrogen2_nucleus,
                       oxygen_atom1
                       ):
        index = 1
        coordinate_nucleus = np.array([0.0, 0.0, 0.0])
        coordinate = np.array([1.0, 1.0, 1.0])
        charge = 2.0
        nucleus = particle.Nucleus(index, coordinate_nucleus, charge)
        potential = nucleus.potential(coordinate)
        assert isinstance(potential, (float, np.ndarray))  # Checking the returned type
        assert oxygen_nucleus.potential(coordinate=oxygen_atom1.coordinate) \
               == pytest.approx(0.75202669489040475, 1e-9)
        assert hydrogen1_nucleus.potential(coordinate=hydrogen2_nucleus.coordinate,
                                           pot_derivative_order=0,
                                           origin_derivative_order=0) \
               == pytest.approx(0.34335113566514336, 1e-9)
        assert hydrogen2_nucleus.potential(coordinate=hydrogen1_nucleus.coordinate,
                                           pot_derivative_order=0,
                                           origin_derivative_order=0) \
               == pytest.approx(0.34335113566514336, 1e-9)
        with pytest.raises(ValueError, match="r_a and r_b cannot be equal."):
            hydrogen1_nucleus.potential(coordinate=hydrogen1_nucleus.coordinate,
                                        pot_derivative_order=0,
                                        origin_derivative_order=0) \
                # Test 1st order derivative
        ref = np.array([-6.3679443173877762e-2, -3.0699045537920477e-2, 4.1315647410858666e-6])
        assert np.allclose(oxygen_nucleus.potential(coordinate=oxygen_atom1.coordinate,
                                                    pot_derivative_order=1), ref)
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
        with pytest.raises(ValueError, match="r_a and r_b cannot be equal."):
            hydrogen1_nucleus.potential(coordinate=hydrogen1_nucleus.coordinate,
                                        pot_derivative_order=0,
                                        origin_derivative_order=1)


class TestVirtualParticle:
    def test_init_virtual_particle(self,
                                   phys_constants
                                   ):
        virtual_particle = particle.VirtualParticle(index=0, coordinate=np.array([1.2919875, 2.156584, -0.0007825])
                                                                        / phys_constants.bohr2angstroms)
        assert virtual_particle.index == 0
        assert virtual_particle.coordinate.shape == (3,)
