"""Tests PyFraME.embedding.subsystem.py"""
import sys, io, pytest
import numpy as np
from pyframe.embedding import subsystem


class TestQuantumSubsystem:
    @pytest.fixture(autouse=True)
    def setup(self, act_wat):
        self.core, self.env = act_wat
    def test_init_with_valid_arguments(self):
        nuclei = self.core.nuclei  # Provide appropriate Nuclei instances for testing
        dens_mat = self.core.density_matrix  # Provide appropriate DensityMatrix instance for testing
        quantum_fragments = self.core.quantum_fragments  # Provide appropriate QuantumFragments instances for testing
        name = "TestQuantumSubsystem"
        q_subsystem = subsystem.QuantumSubsystem(nuclei=nuclei, dens_mat=dens_mat, quantum_fragments=quantum_fragments, name=name)
        assert q_subsystem.name == name
        assert q_subsystem.nuclei == nuclei
        assert q_subsystem.density_matrix == dens_mat
        assert q_subsystem.quantum_fragments == quantum_fragments
        assert q_subsystem.num_nuclei == len(nuclei)
        assert q_subsystem.coordinates.shape == (q_subsystem.num_nuclei, 3)

    def test_init_with_empty_nuclei(self):
        with pytest.raises(ValueError, match="QuantumSubsystem must have at least one Nucleus."):
            subsystem.QuantumSubsystem(nuclei=[], dens_mat=self.core.density_matrix)


    def test_static_potential(self):
        ref_pot_array = np.array([1.9344114763470204, 2.034614421586786,
                         1.8647970050316007, 3.0783131716387415,
                         3.045876112078533, 3.2202498546909775])
        ref_field_array = np.array([[0.00392044, -0.05293017, -0.10427627],
                                    [ 0.00469095, -0.05492494, -0.11713605],
                                    [ 0.00066334, -0.04590806, -0.09858109],
                                    [-0.26160936, 0.11023125, 0.08629496],
                                    [-0.26141526, 0.0838028, 0.09513431],
                                    [-0.27319751, 0.13408809, 0.1133901 ]])
        for i, coordinates in enumerate(self.env.coordinates):
            assert ref_pot_array[i] == self.core.static_potential(coordinate=coordinates,
                                                         pot_derivative_order=0,
                                                         origin_derivative_order=0,
                                                         coord_multipole_order=0)
            assert np.allclose(ref_field_array[i], self.core.static_potential(coordinate=coordinates,
                                                                         pot_derivative_order=1,
                                                                         origin_derivative_order=0,
                                                                         coord_multipole_order=0))


    def test_compute_nuclear_fields(self):
        ref_array = np.array([[0.00392044, -0.05293017, -0.10427627],
                              [0.00469095, -0.05492494, -0.11713605],
                              [0.00066334, -0.04590806, -0.09858109],
                              [-0.26160936, 0.11023125, 0.08629496],
                              [-0.26141526, 0.0838028 , 0.09513431],
                              [-0.27319751, 0.13408809, 0.1133901 ]])
        assert np.allclose(self.core.compute_nuclear_fields(self.env.coordinates), ref_array)
        assert np.all(np.isnan(self.core.compute_nuclear_fields(self.core.coordinates)) == True)


    def test_compute_electric_fields(self,
                                     act_wat_electric_fields,
                                     dummy_integral_driver_factory
                                     ):
        int_driver = dummy_integral_driver_factory(act_wat_electric_fields)
        el_field = self.core.compute_electric_fields(coordinates= self.env.coordinates,
                                                integral_drv=int_driver)
        assert np.allclose(el_field, int_driver.ref_data)

    def test_update_density(self):
        old_density = self.core.density_matrix.density
        new_density = np.eye(3)
        self.core.update_density(new_density)
        assert np.allclose(new_density, self.core.density_matrix.density)
        assert not np.array_equal(old_density, self.core.density_matrix.density)

class TestClassicalSubsystem:
    @pytest.fixture(autouse=True)
    def setup(self, act_wat):
        self.core, self.env = act_wat
    def test_init_with_valid_arguments(self):
        fragments = self.env.classical_fragments
        name = "TestSubsystem"
        c_subsystem = subsystem.ClassicalSubsystem(classical_fragments=fragments, name=name)
        assert c_subsystem.name == name
        assert c_subsystem.classical_fragments == fragments
        assert c_subsystem.num_atoms == 6
        assert c_subsystem.coordinates.shape == (c_subsystem.num_atoms, 3)
        assert c_subsystem.polarizabilities.shape == (c_subsystem.num_atoms, 3, 3)
        assert c_subsystem.indices.shape == (c_subsystem.num_atoms,)
        assert c_subsystem.exclusions == [(0, 1, 2), (0, 1, 2), (0, 1, 2), (3, 4, 5), (3, 4, 5), (3, 4, 5)]
        assert c_subsystem.induced_dipoles.shape == (c_subsystem.num_atoms, 3)
        assert c_subsystem.multipole_fields.shape == (c_subsystem.num_atoms, 3)

    def test_init_with_empty_classical_fragments(self):
        with pytest.raises(ValueError, match="ClassicalSubsystem must have at least one ClassicalFragment"):
            subsystem.ClassicalSubsystem(classical_fragments=[])


    def test_self_energy(self,
                         wat_wat,
                         acrolein_wat
                         ):
        assert self.env.self_energy() == -2.2376361011309555e-05
        core_wat, env_wat = wat_wat
        assert env_wat.self_energy() == -0.00718198734326498
        # value tested against dalton and pelib
        core_ac, env_ac = acrolein_wat
        assert pytest.approx(0.001012591928, abs=1e-9) == env_ac.self_energy()


    def test_static_potential(self):
        ref_pot = np.array([0.00158312, 0.00097756, 0.00153459,
                            0.00216101, 0.00095463, 0.00092537,
                            0.00054562, 0.0012971, 0.00155222,
                            0.00170403])
        ref_field = np.array([[-3.57871685e-05, -6.08385213e-05, -4.16530095e-04],
                              [-2.68149742e-04,  4.86543506e-05, -3.00432016e-04],
                              [ 6.23813670e-05, -7.74281397e-05, -2.51253113e-04],
                              [ 8.65448666e-05, -1.40820676e-04, -6.35728002e-04],
                              [-2.30570996e-04,  4.57958753e-05, -2.72489018e-04],
                              [-2.46673973e-04,  5.46186536e-05, -1.37596052e-04],
                              [-0.00052421,  0.00014397, -0.00039426],
                              [ 2.67417460e-05, -6.70011124e-05, -1.47749387e-04],
                              [ 7.84632264e-05, -5.54709547e-05, -2.38816559e-04],
                              [ 0.00014484, -0.0001176 , -0.00025409]])
        ref_field_deriv = np.array(
            [[-9.69852521e-05,  1.70233222e-05, -1.29555083e-04, -9.70810591e-06, 6.33704988e-05, 1.06693358e-04],
            [-1.60077407e-04,  6.96123686e-05, -7.39728226e-05, -1.33963213e-05, 5.05605259e-05, 1.73473728e-04],
            [-4.03908839e-05,  1.41900109e-06, -7.99330085e-05, -2.05042679e-05, 3.42781058e-05, 6.08951518e-05],
            [-8.29657867e-05, -1.42127811e-05, -2.16956071e-04,  4.06983586e-06, 9.94223097e-05, 7.88959508e-05],
            [-1.25008276e-04,  8.21396573e-05, -6.71466847e-05, -3.63421262e-05, 4.33944129e-05, 1.61350402e-04],
            [-1.38184163e-04,  5.06788325e-05, -3.85966621e-05, -1.75111041e-05, 3.53465208e-05, 1.55695267e-04],
            [-2.69440850e-04,  1.26287375e-04, -6.11783426e-05,  1.27560083e-05, 5.94434260e-05, 2.56684841e-04],
            [-4.54833255e-05,  1.00924511e-05, -4.81027873e-05, -2.66102992e-05, 2.34740318e-05, 7.20936248e-05],
            [-3.32816075e-05, -2.77890089e-06, -7.86157006e-05, -2.17500263e-05, 2.83560955e-05, 5.50316338e-05],
            [-8.57569968e-06, -1.22453942e-05, -7.85203841e-05, -1.87806155e-05, 3.13399924e-05,  2.73563152e-05]])
        for i, coordinates in enumerate(self.core.coordinates):
            assert pytest.approx(ref_pot[i], abs=1e-8) == self.env.static_potential(coordinate=coordinates,
                                                                               pot_derivative_order=0,
                                                                               origin_derivative_order=0,
                                                                               coord_multipole_order=0,
                                                                               array_of_potentials=False)[0]
            assert  np.allclose(ref_field[i], self.env.static_potential(coordinate=coordinates,
                                                                   pot_derivative_order=1,
                                                                   origin_derivative_order=0,
                                                                   coord_multipole_order=0,
                                                                   array_of_potentials=False),
                                atol=1e-8)

            assert  np.allclose(ref_field_deriv[i], self.env.static_potential(coordinate=coordinates,
                                                          pot_derivative_order=2,
                                                          origin_derivative_order=0,
                                                          coord_multipole_order=0,
                                                          array_of_potentials=False),
                                atol=1e-8)


    def test_solve_induced_dipoles(self,
                                   act_wat_electric_fields,
                                   dummy_integral_driver_factory
                                   ):
        int_driver = dummy_integral_driver_factory(act_wat_electric_fields)
        electric_field = self.core.compute_electric_fields(coordinates=self.env.coordinates, integral_drv=int_driver)
        nuclear_field = self.core.compute_nuclear_fields(coordinates=self.env.coordinates)
        self.env.solve_induced_dipoles(external_fields=(nuclear_field + electric_field), threshold=1e-10)
        ref_dipoles = np.array([[-0.05130705, -0.25165223, -0.4511121 ],
                                [0.00465468, -0.27383291, -0.18815435],
                                [-0.09539097, 0.03976932, -0.19509574],
                                [-1.17424447, 0.51284179, 0.50156209],
                                [-0.14716566, -0.03273562, 0.12889495],
                                [-0.44555632, 0.00482074, -0.09598545]])
        assert np.allclose(self.env.induced_dipoles.induced_dipoles, ref_dipoles)
        assert self.env.induced_dipoles.number_of_iterations == 4
        captured_output = io.StringIO()
        sys.stdout = captured_output
        self.env.solve_induced_dipoles(external_fields=(nuclear_field + electric_field), threshold=1e-10)
        sys.stdout = sys.__stdout__
        assert ("Residue norm between new and old external fields is 0, induced dipoles will not be recalculated.\n" ==
                captured_output.getvalue())
        captured_output = io.StringIO()
        sys.stdout = captured_output
        self.env.solve_induced_dipoles(external_fields=(nuclear_field + electric_field + np.full(electric_field.shape,
                                                                                            1e-10)), threshold=1e-10)
        sys.stdout = sys.__stdout__
        assert ("Residue norm between new and old external fields is smaller than 1e-6, old induced dipoles will be"
                          " used as a starting guess.\n" == captured_output.getvalue())
        assert self.env.induced_dipoles.number_of_iterations == 2
        captured_output = io.StringIO()
        sys.stdout = captured_output
        self.env.solve_induced_dipoles(external_fields=(nuclear_field + electric_field + np.full(electric_field.shape,
                                                                                            0.1)), threshold=1e-10)
        sys.stdout = sys.__stdout__
        assert ("Residue norm between new and old external fields is larger than 1e-6, old induced dipoles will "
                          "not be used as a starting guess.\n" == captured_output.getvalue())


    def test_induced_dipoles_dataclass(self):
        induced_dipoles = np.full((3, 3), 1)
        external_fields = np.full((3, 3), 2)
        induced_dipole_fields = np.full((3, 3), 3)
        num_iter = 1
        data = subsystem.InducedDipoles(induced_dipoles=induced_dipoles,
                                        external_fields=external_fields,
                                        induced_dipole_fields=induced_dipole_fields,
                                        number_of_iterations=num_iter,
                                        solver='test_solver')
        assert np.allclose(data.induced_dipoles, induced_dipoles)
        assert np.allclose(data.external_fields, external_fields)
        assert np.allclose(data.induced_dipole_fields, induced_dipole_fields)
        assert np.allclose(data.number_of_iterations, num_iter)
        assert data.solver == 'test_solver'
