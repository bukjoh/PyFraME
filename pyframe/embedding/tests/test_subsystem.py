"""Tests PyFraME.embedding.subsystem.py"""
import os, sys, io, pytest
import numpy as np
from pyframe.embedding import read_input, subsystem
from qcelemental import PhysicalConstantsContext

constants = PhysicalConstantsContext('CODATA2018')
core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_test.json')

class TestQuantumSubsystem:
    def test_init_with_valid_arguments(self):
        nuclei = core.nuclei  # Provide appropriate Nuclei instances for testing
        dens_mat = core.density_matrix  # Provide appropriate DensityMatrix instance for testing
        quantum_fragments = core.quantum_fragments  # Provide appropriate QuantumFragments instances for testing
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
            subsystem.QuantumSubsystem(nuclei=[], dens_mat=core.density_matrix)


    def test_quantum_static_potential(self):
        ref_pot_array = np.array([1.9344114763470204, 2.034614421586786,
                         1.8647970050316007, 3.0783131716387415,
                         3.045876112078533, 3.2202498546909775])
        ref_field_array = np.array([[0.00392044, -0.05293017, -0.10427627],
                                    [ 0.00469095, -0.05492494, -0.11713605],
                                    [ 0.00066334, -0.04590806, -0.09858109],
                                    [-0.26160936, 0.11023125, 0.08629496],
                                    [-0.26141526, 0.0838028, 0.09513431],
                                    [-0.27319751, 0.13408809, 0.1133901 ]])
        for i, coordinates in enumerate(env.coordinates):
            assert ref_pot_array[i] == core.static_potential(coordinate=coordinates,
                                                         pot_derivative_order=0,
                                                         origin_derivative_order=0,
                                                         coord_multipole_order=0)
            assert np.allclose(ref_field_array[i], core.static_potential(coordinate=coordinates,
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
        assert np.allclose(core.compute_nuclear_fields(env.coordinates), ref_array)
        assert np.all(np.isnan(core.compute_nuclear_fields(core.coordinates)) == True)


    def test_compute_electric_fields(self):
        int_driver = DummyIntegralDriver()
        el_field = core.compute_electric_fields(coordinates= env.coordinates,
                                                integral_drv=int_driver)
        assert np.allclose(el_field, int_driver.ref_data)

    def test_update_density(self):
        old_density = core.density_matrix.density
        new_density = np.eye(3)
        core.update_density(new_density)
        assert np.allclose(new_density, core.density_matrix.density)
        assert not np.array_equal(old_density, core.density_matrix.density)


class TestClassicalSubsystem:
    def test_init_with_valid_arguments(self):
        fragments = env.classical_fragments
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


    def test_self_energy(self):
        # TODO test again lsdalton etc.
        assert env.self_energy() == -2.2376361011309555e-05
        core_wat, env_wat = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/wat_in_wat_test.json')
        assert env_wat.self_energy() == -0.00718198734326498
# ToDO
    def test_static_potential(self):
        return


    def test_solve_induced_dipoles(self):
        int_driver = DummyIntegralDriver()
        electric_field = core.compute_electric_fields(coordinates=env.coordinates, integral_drv=int_driver)
        nuclear_field = core.compute_nuclear_fields(coordinates=env.coordinates)
        env.solve_induced_dipoles(external_fields=(nuclear_field + electric_field), threshold=1e-10)
        ref_dipoles = np.array([[-0.05130705, -0.25165223, -0.4511121 ],
                                [0.00465468, -0.27383291, -0.18815435],
                                [-0.09539097, 0.03976932, -0.19509574],
                                [-1.17424447, 0.51284179, 0.50156209],
                                [-0.14716566, -0.03273562, 0.12889495],
                                [-0.44555632, 0.00482074, -0.09598545]])
        assert np.allclose(env.induced_dipoles.induced_dipoles, ref_dipoles)
        assert env.induced_dipoles.number_of_iterations == 4
        captured_output = io.StringIO()
        sys.stdout = captured_output
        env.solve_induced_dipoles(external_fields=(nuclear_field + electric_field), threshold=1e-10)
        sys.stdout = sys.__stdout__
        assert ("Residue norm between new and old external fields is 0, induced dipoles will not be recalculated.\n" ==
                captured_output.getvalue())
        captured_output = io.StringIO()
        sys.stdout = captured_output
        env.solve_induced_dipoles(external_fields=(nuclear_field + electric_field + np.full(electric_field.shape,
                                                                                            1e-10)), threshold=1e-10)
        sys.stdout = sys.__stdout__
        assert ("Residue norm between new and old external fields is smaller than 1e-6, old induced dipoles will be"
                          " used as a starting guess.\n" == captured_output.getvalue())
        assert env.induced_dipoles.number_of_iterations == 2
        captured_output = io.StringIO()
        sys.stdout = captured_output
        env.solve_induced_dipoles(external_fields=(nuclear_field + electric_field + np.full(electric_field.shape,
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
                                            number_of_iterations=num_iter)
            assert np.allclose(data.induced_dipoles, induced_dipoles)
            assert np.allclose(data.external_fields, external_fields)
            assert np.allclose(data.induced_dipole_fields, induced_dipole_fields)
            assert np.allclose(data.number_of_iterations, num_iter)


class DummyIntegralDriver:
    def __init__(self):
        self.ref_data = np.array([[-0.02000067, -0.01409951, -0.00842215],
                                  [-0.01898043, -0.01393681, -0.0083458],
                                  [-0.02048344, -0.01486784, -0.00814858],
                                  [-0.00587545, -0.0102123, -0.00982018],
                                  [-0.00592655, -0.01018911, -0.01021058],
                                  [-0.00584855, -0.00979238, -0.00949128]])
        self.coordinates = None
        self.density = None
    def electric_fields(self,
                        coordinates: np.ndarray,
                        density: np.ndarray):
        self.coordinates = coordinates
        self.density = density
        return self.ref_data
