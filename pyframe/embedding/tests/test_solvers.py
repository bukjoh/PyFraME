"""Tests PyFraME.embedding.solvers.py"""
import pytest
import numpy as np
import os

from pyframe.embedding import read_input
from pyframe.embedding.solvers import induced_dipoles_jacobi
from mpi4py import MPI


def example_data(data):
    return data.coordinates, data.polarizabilities, data.exclusions, data.indices, data.multipole_fields


def test_induced_dipoles_jacobi(
        act_wat,
        act_wat_electric_fields
):
    ref_dipoles = np.array([[-0.05130705, -0.25165223, -0.4511121],
                            [0.00465468, -0.27383291, -0.18815435],
                            [-0.09539097, 0.03976932, -0.19509574],
                            [-1.17424447, 0.51284179, 0.50156209],
                            [-0.14716566, -0.03273562, 0.12889495],
                            [-0.44555632, 0.00482074, -0.09598545]])
    coordinates, polarizabilities, exclusions, indices, multipole_fields = example_data(act_wat[1])
    electric_field = act_wat_electric_fields
    nuclear_field = np.array([[0.00392044, -0.05293017, -0.10427627],
                              [0.00469095, -0.05492494, -0.11713605],
                              [0.00066334, -0.04590806, -0.09858109],
                              [-0.26160936, 0.11023125, 0.08629496],
                              [-0.26141526, 0.0838028, 0.09513431],
                              [-0.27319751, 0.13408809, 0.1133901]])
    fields = electric_field + nuclear_field + multipole_fields
    starting_guess = np.zeros([len(fields), 3])
    for i, field in enumerate(fields):
        starting_guess[i, :] = np.einsum('ij, j', polarizabilities[i], field)
    threshold = 1e-10
    ind_dipoles, iteration = induced_dipoles_jacobi(coordinates, polarizabilities, exclusions, indices,
                                                    fields, starting_guess, threshold)
    # Test if output has the correct dipoles
    assert np.allclose(ind_dipoles, ref_dipoles)
    # Test if the output has the correct shape
    assert ind_dipoles.shape == (6, 3)
    # Test if the iteration count is reasonable
    assert iteration < 500  # arbitrary upper limit for iteration
    # Test for different threshold values
    thresholds = [1e-10, 1e-12, 1e-15]
    for threshold in thresholds:
        _, iteration = induced_dipoles_jacobi(coordinates, polarizabilities, exclusions, indices,
                                              fields, starting_guess, threshold)
        assert iteration < 500  # arbitrary upper limit for iteration


def test_induced_dipoles_jacobi_edge_cases():
    # Test with minimum input size
    coordinates = np.ones([1, 3])
    polarizabilities = np.ones([1, 3, 3])
    exclusions = [[1]]
    indices = np.ones(1)
    fields = np.ones([1, 3])
    starting_guess = np.ones([1, 3])
    threshold = 1e-10
    ind_dipoles, _ = induced_dipoles_jacobi(coordinates, polarizabilities, exclusions, indices,
                                            fields, starting_guess, threshold)
    assert np.allclose(ind_dipoles, 3 * np.ones([1, 3]))


def test_induced_dipoles_jacobi_invalid_inputs():
    # Test with invalid inputs
    with pytest.raises(ValueError, match="Wrong input format."):
        induced_dipoles_jacobi(None,
                               None,
                               None,
                               None,
                               None,
                               None,
                               None)


def test_induced_dipoles_jacobi_stability():
    coordinates = np.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]])
    polarizabilities = np.array([[[.1, .1, .1], [.1, .1, .1], [.1, .1, .1]],
                                 [[.1, .1, .1], [.1, .1, .1], [.1, .1, .1]],
                                 [[.1, .1, .1], [.1, .1, .1], [.1, .1, .1]]])
    exclusions = [[i] for i in range(3)]
    indices = np.array([0., 1., 2.])
    fields = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    starting_guess = np.ones([3, 3])
    threshold = 1e-6
    ref_dipoles = np.array([[0.11632816, 0.11632816, 0.11632816],
                            [0.12686485, 0.12686485, 0.12686485],
                            [0.11632816, 0.11632816, 0.11632816]])
    # Call the function multiple times with the same inputs
    for i in range(5):
        ind_dipoles, num_iter = induced_dipoles_jacobi(coordinates, polarizabilities, exclusions, indices,
                                                       fields, starting_guess, threshold)
        assert np.allclose(ind_dipoles, ref_dipoles)  # Assert that the output is consistent


@pytest.mark.mpi()
def test_induced_dipoles_jacobi_large_inputs():
    import time
    # Generate large input arrays
    for size in [1, 10, 30]:
        mu, sigma = 0.0, 0.1
        coordinates = np.random.normal(mu, sigma, size=(size, 3))
        polarizabilities = np.random.normal(mu, sigma, size=(size, 3, 3))
        exclusions = [[i] for i in range(size)]
        indices = np.arange(size)
        fields = np.random.normal(mu, sigma, size=(size, 3))
        starting_guess = np.ones([size, 3])
        threshold = 1e-6
        # Measure execution time
        start_time = time.time()
        induced_dipoles_jacobi(coordinates, polarizabilities, exclusions, indices,
                               fields, starting_guess, threshold)
        end_time = time.time()
        # Assert that the execution time is reasonable
        assert end_time - start_time < size  # Adjust the time limit based on your performance requirements

    comm = MPI.COMM_WORLD
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_big.json', comm=comm)
    print("Calculate induced dipoles without external fields.")
    start_time = time.time()
    env.solve_induced_dipoles()
    end_time = time.time()
    print("Execution time:", end_time - start_time)
    gathered_summed_arr = comm.gather(env.induced_dipoles.induced_dipoles, root=0)
    gathered_summed_arr2 = comm.gather(env._multipole_fields, root=0)
    # Check if the gathered arrays are equal on all processes
    if comm.Get_rank() == 0:
        for i in range(1, comm.Get_size()):
            assert np.array_equal(gathered_summed_arr[0],
                                  gathered_summed_arr[i]), "Arrays are not equal across processes"
            assert np.array_equal(gathered_summed_arr2[0],
                                  gathered_summed_arr2[i]), "Arrays are not equal across processes"
    MPI.Finalize()
