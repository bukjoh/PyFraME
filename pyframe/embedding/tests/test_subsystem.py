"""Tests PyFraME.embedding.subsystem.py"""
import pytest
import numpy as np

from pyframe.embedding import subsystem
from pyframe.embedding.logging_util import log_manager


class TestQuantumSubsystem:
    @pytest.fixture(autouse=True)
    def setup(self, act_wat, two_oxygen, two_wat):
        self.core, self.env = act_wat
        self.core_oxygen, self.env_oxygen = two_oxygen
        self.core_two_wat, self.env_two_wat = two_wat

    def test_init_with_valid_arguments(self):
        nuclei = self.core.nuclei  # Provide appropriate Nuclei instances for testing
        quantum_fragments = self.core.quantum_fragments  # Provide appropriate QuantumFragments instances for testing
        name = "TestQuantumSubsystem"
        q_subsystem = subsystem.QuantumSubsystem(nuclei=nuclei, quantum_fragments=quantum_fragments,
                                                 name=name)
        assert q_subsystem.name == name
        assert q_subsystem.nuclei == nuclei
        assert q_subsystem.quantum_fragments == quantum_fragments
        assert q_subsystem.num_nuclei == len(nuclei)
        assert q_subsystem.coordinates.shape == (q_subsystem.num_nuclei, 3)

    def test_init_with_no_nuclei(self):
        c_subsystem = subsystem.QuantumSubsystem(nuclei=None)
        assert c_subsystem.num_nuclei == 0

    def test_compute_nuclear_fields(self):
        ref_array = np.array([[0.00392044, -0.05293017, -0.10427627],
                              [0.00469095, -0.05492494, -0.11713605],
                              [0.00066334, -0.04590806, -0.09858109],
                              [-0.26160936, 0.11023125, 0.08629496],
                              [-0.26141526, 0.0838028, 0.09513431],
                              [-0.27319751, 0.13408809, 0.1133901]])
        assert np.allclose(self.core.compute_nuclear_fields(self.env.coordinates), ref_array)
        # with pytest.raises(ValueError, match="r_a and r_b cannot be equal."):
        #     self.core.compute_nuclear_fields(self.core.coordinates)

    def test_compute_nuclear_field_gradients(self):
        ref_grads = np.array([9.5311929756771267e-3, 7.7985188788112876e-3, -1.0495468203592808e-6,
                              -2.8858133904975198e-3, -5.0597310570715463e-7, -6.6453795851796043e-3], dtype=np.float64)
        # self.core_oxygen.compute_nuclear_field_gradients(self.env_oxygen.coordinates)
        assert np.allclose(ref_grads, self.core_oxygen.compute_nuclear_field_gradients(self.env_oxygen.coordinates))

        ref_grads = np.array([[[9.53119298e-03, 7.79851888e-03, -1.04954682e-06,
                                -2.88581339e-03, -5.05973103e-07, -6.64537959e-03],
                               [1.64767082e-02, 1.40353017e-02, -2.73260760e-06,
                                -4.74648718e-03, -1.35970036e-06, -1.17302210e-02],
                               [5.66827354e-03, 7.62087970e-03, 1.02387457e-05,
                                -2.21554931e-04, 7.02008439e-06, -5.44671861e-03]],

                              [[2.17297931e-03, 1.65026283e-03, -9.27777498e-07,
                                -7.11799614e-04, -4.21301494e-07, -1.46117969e-03],
                               [4.31823566e-03, 3.38501097e-03, -2.51717115e-06,
                                -1.37061531e-03, -1.17269747e-06, -2.94762035e-03],
                               [1.17631179e-03, 1.62630080e-03, 2.15048995e-06,
                                -2.09897761e-05, 1.49995275e-06, -1.15532202e-03]],

                              [[1.15981561e-03, 5.67359046e-04, 3.31949401e-08,
                                -4.91851181e-04, 1.03040022e-08, -6.67964432e-04],
                               [1.98882757e-03, 9.09798164e-04, 1.91525993e-08,
                                -8.61596532e-04, 5.59199997e-09, -1.12723104e-03],
                               [8.44935406e-04, 6.96866735e-04, 1.26731929e-06,
                                -2.53431261e-04, 6.14820143e-07, -5.91504145e-04]]])
        # FIXME ist einmal im test gefailed???
        assert np.allclose(ref_grads, self.core_two_wat.compute_nuclear_field_gradients(self.env_two_wat.coordinates))

    def test_compute_electric_fields(self,
                                     act_wat_electric_fields,
                                     act_wat_density_matrix,
                                     dummy_integral_driver_factory
                                     ):
        int_driver = dummy_integral_driver_factory(act_wat_electric_fields)
        el_field = self.core.compute_electronic_fields(coordinates=self.env.coordinates,
                                                       density_matrix=act_wat_density_matrix,
                                                       integral_driver=int_driver)
        assert np.allclose(el_field, int_driver.ref_data)


class TestClassicalSubsystem:
    @pytest.fixture(autouse=True)
    def setup(self,
              act_wat,
              wat_wat,
              acrolein_wat,
              act_wat_big,
              butadiene_water
              ):
        self.core_act_t, self.env_act_t = act_wat
        self.core_act, self.env_act = act_wat_big
        self.core_wat, self.env_wat = wat_wat
        self.core_ac, self.env_ac = acrolein_wat
        self.core_but, self.env_but = butadiene_water

    def test_init_with_valid_arguments(self):
        fragments = self.env_act_t.classical_fragments
        name = "TestSubsystem"
        c_subsystem = subsystem.ClassicalSubsystem(classical_fragments=fragments, name=name)
        assert c_subsystem.name == name
        assert c_subsystem.classical_fragments == fragments
        assert c_subsystem.num_atoms == 6
        assert c_subsystem.coordinates.shape == (c_subsystem.num_atoms, 3)
        assert c_subsystem.dipole_dipole_polarizabilities.shape == (c_subsystem.num_atoms, 3, 3)
        assert c_subsystem.indices.shape == (c_subsystem.num_atoms,)
        assert c_subsystem.exclusions == [(0, 1, 2), (0, 1, 2), (0, 1, 2), (3, 4, 5), (3, 4, 5), (3, 4, 5)]
        assert c_subsystem.induced_dipoles.induced_dipoles.shape == (c_subsystem.num_atoms, 3)
        assert c_subsystem.multipole_fields.shape == (c_subsystem.num_atoms, 3)

    def test_init_with_empty_classical_fragments(self):
        with pytest.raises(ValueError, match="ClassicalSubsystem must have at least one ClassicalFragment"):
            subsystem.ClassicalSubsystem(classical_fragments=[])

    def test_environment_energy(self,
                                wat_wat,
                                acrolein_wat,
                                act_wat_big,
                                act_wat,
                                butadiene_water
                                ):
        assert self.env_act_t.environment_energy == pytest.approx(-2.2376361011309555e-05, abs=1e-12)
        assert self.env_wat.environment_energy == pytest.approx(-0.00718198734326498, abs=1e-12)
        # value tested against dalton with pelib
        assert pytest.approx(0.001012591928, abs=1e-9) == self.env_ac.environment_energy
        # value tested against dalton with pelib
        assert pytest.approx(-5.200360556757, abs=1e-8) == self.env_act.environment_energy

        assert pytest.approx(-2.2376361011313024e-05, abs=1e-12) == self.env_act_t.environment_energy
        assert pytest.approx(-0.0184325879671965, abs=1e-12) == self.env_but.environment_energy

    def test_compute_multipole_fields(self):
        ref_fields = np.array([[-6.88526592e-03, 8.61243977e-03, -3.98239429e-03],
                               [-2.31127305e-02, 3.00417481e-02, -2.51125273e-02],
                               [-1.68499637e-03, 5.00369550e-03, -4.44046917e-03],
                               [-2.46855201e-03, 1.15151596e-02, -1.88504755e-02],
                               [4.51744687e-03, 1.03509735e-03, -9.02918842e-03],
                               [-3.23334755e-03, 1.82938297e-03, -7.10269295e-03],
                               [2.72946110e-04, -4.63074208e-04, -1.84476866e-04],
                               [3.14770478e-04, -5.85885190e-04, -4.73097572e-04],
                               [1.98420210e-04, -6.15280152e-04, -5.16664149e-05]])
        assert np.allclose(self.env_but.multipole_fields, ref_fields)
        ref_fields = np.array([[-6.43939331e-03, -1.75558132e-03, 7.58932558e-04],
                               [-3.49230589e-03, 1.20406697e-03, 5.97577089e-04],
                               [-4.45114544e-03, -1.97804776e-03, 1.43844660e-03],
                               [-1.03355900e-03, 1.32524544e-03, -3.79201566e-03],
                               [-3.04085251e-05, 1.36646363e-03, -2.09178543e-03],
                               [-4.20853287e-03, 1.37263607e-03, -5.81682184e-03],
                               [1.33413235e-03, 1.82510027e-03, 1.62329284e-03],
                               [2.46763060e-03, 4.93517001e-04, 1.69756037e-03],
                               [1.40764452e-03, 1.28334739e-03, 1.36600708e-04],
                               [-3.11205363e-03, -2.15161862e-03, 3.36398086e-03],
                               [-6.70973218e-03, -3.99943622e-03, 3.89476831e-03],
                               [-3.48489739e-03, -1.25625556e-03, 4.05203538e-03]])
        assert np.allclose(self.env_wat.multipole_fields, ref_fields)

    def test_solve_induced_dipoles(self,
                                   act_wat_electric_fields,
                                   act_wat_density_matrix,
                                   dummy_integral_driver_factory
                                   ):
        int_driver = dummy_integral_driver_factory(act_wat_electric_fields)
        electric_field = self.core_act_t.compute_electronic_fields(coordinates=self.env_act_t.coordinates,
                                                                   density_matrix=act_wat_density_matrix,
                                                                   integral_driver=int_driver)
        nuclear_field = self.core_act_t.compute_nuclear_fields(coordinates=self.env_act_t.coordinates)
        self.env_act_t.solve_induced_dipoles(external_fields=(nuclear_field + electric_field), threshold=1e-10)
        ref_dipoles = np.array([[-0.05130705, -0.25165223, -0.4511121],
                                [0.00465468, -0.27383291, -0.18815435],
                                [-0.09539097, 0.03976932, -0.19509574],
                                [-1.17424447, 0.51284179, 0.50156209],
                                [-0.14716566, -0.03273562, 0.12889495],
                                [-0.44555632, 0.00482074, -0.09598545]])
        assert np.allclose(self.env_act_t.induced_dipoles.induced_dipoles, ref_dipoles)
        assert self.env_act_t.induced_dipoles.number_of_iterations == 4
        log_manager.set_level(10)
        self.env_act_t.solve_induced_dipoles(external_fields=(nuclear_field + electric_field), threshold=1e-10)
        assert ("Residue norm between new and old external fields is 0, induced dipoles will not be recalculated."
                in log_manager.get_logs())
        log_manager.reset()
        self.env_act_t.solve_induced_dipoles(
            external_fields=(nuclear_field + electric_field + np.full(electric_field.shape,
                                                                      1e-10)),
            threshold=1e-10)
        assert ("Residue norm between new and old external fields is smaller than 1e-6, old induced dipoles will be"
                " used as a starting guess." in log_manager.get_logs())
        assert self.env_act_t.induced_dipoles.number_of_iterations == 2
        log_manager.reset()
        self.env_act_t.solve_induced_dipoles(
            external_fields=(nuclear_field + electric_field + np.full(electric_field.shape,
                                                                      0.1)),
            threshold=1e-10)
        assert ("Residue norm between new and old external fields is larger than 1e-6, old induced dipoles will "
                "not be used as a starting guess." in log_manager.get_logs())

    def test_induced_dipoles_dataclass(self):
        induced_dipoles = np.full((3, 3), 1)
        external_fields = np.full((3, 3), 2)
        num_iter = 1
        data = subsystem.InducedDipoles(induced_dipoles=induced_dipoles,
                                        external_fields=external_fields,
                                        number_of_iterations=num_iter,
                                        solver='test_solver',
                                        threshold=1e-8)
        assert np.allclose(data.induced_dipoles, induced_dipoles)
        assert np.allclose(data.external_fields, external_fields)
        assert np.allclose(data.number_of_iterations, num_iter)
        assert data.solver == 'test_solver'
        assert data.threshold == 1e-8
