"""Tests PyFraME.embedding.solvers.py"""
import pytest
import numpy as np
import os

from pyframe.embedding import read_input
from pyframe.embedding.solvers import (induced_dipoles_jacobi, induced_dipoles_jidiis, induced_dipoles_dcji,
                                       induced_dipoles_dcjidiis, divide_and_conquer,
                                       direct_inversion_iterative_subspace, kmeans_clustering,
                                       induced_dipoles_fmm)
from mpi4py import MPI


def example_data(data):
    return data.coordinates, data.dipole_dipole_polarizabilities, data.exclusions, data.indices, data.multipole_fields


def test_induced_dipoles_fmm(
        act_wat,
        act_wat_electric_fields,
        act_wat_2
):
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
    mic = False
    box = np.array([])
    threshold = 1e-8
    ind_dipoles, iteration = induced_dipoles_fmm(coordinates, polarizabilities, exclusions, indices,
                                                 fields, starting_guess, mic, box, threshold)
    ref_dipoles = np.array([[-0.05130705, -0.25165223, -0.4511121],
                            [0.00465468, -0.27383291, -0.18815435],
                            [-0.09539097, 0.03976932, -0.19509574],
                            [-1.17424447, 0.51284179, 0.50156209],
                            [-0.14716566, -0.03273562, 0.12889495],
                            [-0.44555632, 0.00482074, -0.09598545]])
    assert np.allclose(ind_dipoles, ref_dipoles)

    box, core, env = act_wat_2
    nuc_fields = core.compute_nuclear_fields(env.coordinates)
    el_fields = np.array([[0.0814927, 0.39995827, -0.18840381],
                          [-0.00149233, 0.49744207, -0.2404297],
                          [0.05975826, 0.3635538, -0.09437206],
                          [0.0053478, -0.10459346, 0.33557309],
                          [-0.01308861, -0.09001337, 0.24563423],
                          [-0.06246361, -0.14415668, 0.42036034],
                          [-0.09056117, 0.19735866, 0.31485522],
                          [-0.14647283, 0.14676371, 0.35576568],
                          [-0.10172189, 0.18597394, 0.225218],
                          [-0.2164909, 0.2726447, 0.11600892],
                          [-0.15768918, 0.30811943, 0.10392172],
                          [-0.18340442, 0.21626795, 0.05592591],
                          [0.42193843, 0.12241386, -0.03793133],
                          [0.48336731, 0.12286773, 0.06612851],
                          [0.29090074, 0.10965616, -0.01829478],
                          [0.22201488, -0.21833702, -0.30330785],
                          [0.32127524, -0.16531516, -0.35813159],
                          [0.18413327, -0.16766409, -0.19219296],
                          [0.3938038, -0.02816132, -0.19394907],
                          [0.43172025, 0.0125735, -0.12447215],
                          [0.35588472, 0.03144, -0.25581068],
                          [-0.30327181, -0.2031279, -0.53012904],
                          [-0.14859804, -0.11926033, -0.47783922],
                          [-0.37851819, -0.06672782, -0.53004169],
                          [-0.30956041, -0.21057398, 0.50207069],
                          [-0.41881029, -0.09511637, 0.52329293],
                          [-0.271854, -0.19640153, 0.32659814],
                          [0.1154456, 0.10504699, 0.36580086],
                          [0.08599618, 0.03104651, 0.38459685],
                          [0.04796868, 0.15788475, 0.37857585],
                          [-0.24190075, -0.23442675, -0.11535178],
                          [-0.24355125, -0.25974957, -0.20416162],
                          [-0.3487222, -0.26524085, -0.09504192],
                          [-0.58441504, -0.20558466, -0.03889671],
                          [-1.08167379, -0.3521555, 0.02578574],
                          [-0.46100817, -0.0993336, 0.04824673],
                          [-0.35977966, 0.10753524, 0.29962407],
                          [-0.30855095, 0.05334029, 0.19937811],
                          [-0.3425848, 0.19851887, 0.24107572],
                          [-0.27607349, 0.52777608, -0.25822781],
                          [-0.25680746, 0.36588248, -0.12297735],
                          [-0.58942081, 0.81252454, -0.35565372],
                          [0.26348549, 0.19612683, -0.24731776],
                          [0.21694023, 0.26938484, -0.2578684],
                          [0.32298631, 0.20666237, -0.18935843]])

    env.solve_induced_dipoles(threshold=1e-8, solver='fmm', mic=False,
                              external_fields=(-1 * el_fields + nuc_fields))
    ind_dip_fmm = env.induced_dipoles.induced_dipoles
    env.solve_induced_dipoles(threshold=1e-8, solver='jacobi', mic=False,
                              external_fields=(-1 * el_fields + nuc_fields))
    ind_dip_ji = env.induced_dipoles.induced_dipoles
    assert np.allclose(ind_dip_fmm, ind_dip_ji)

    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json')
    env.solve_induced_dipoles(threshold=1e-8, solver='fmm')
    ref_ind_dip = np.array([[-5.33083758e-03, 1.80899242e-02, -4.67124873e-02],
                            [-6.25853096e-03, 7.25840460e-03, -1.03916985e-02],
                            [-2.48028626e-02, 2.09508571e-02, -5.74245555e-02],
                            [-4.03295102e-02, -2.81401191e-02, -3.81210336e-02],
                            [-4.06230086e-02, -1.38089059e-02, -4.50891871e-02],
                            [-4.09294307e-05, -7.71654933e-03, -1.10062634e-02],
                            [-1.85295339e-02, 2.86513856e-02, -6.41668069e-03],
                            [-1.89700263e-02, 3.32285459e-02, -1.05088196e-02],
                            [-4.44393262e-03, 6.60779904e-03, -6.06423064e-03],
                            [3.35370357e-02, -1.38589706e-02, -1.33829243e-01],
                            [8.46508800e-03, -1.95121184e-02, -4.13855335e-02],
                            [9.41524338e-02, -7.63125728e-03, -1.05125435e-01],
                            [-3.51664680e-02, -9.46800752e-02, 3.21263242e-02],
                            [-3.05118301e-02, -5.72413535e-02, -7.15883504e-03],
                            [-6.53756795e-03, -1.78294688e-02, 1.00302692e-02],
                            [-5.47042853e-02, -2.59646233e-02, 7.46069721e-02],
                            [-1.21777884e-02, -7.14485835e-03, 1.14801021e-02],
                            [-4.89534262e-03, 8.89451393e-04, 3.05621760e-02],
                            [-2.45470732e-02, 3.50936051e-02, 1.62710897e-02],
                            [-7.08697146e-03, 3.29550988e-02, 2.01553381e-02],
                            [-3.98872500e-02, 2.68723152e-03, 2.96662818e-02],
                            [1.58478302e-01, 1.75607576e-02, -9.62387317e-02],
                            [3.13149664e-02, 1.50598941e-02, 3.26735356e-03],
                            [1.74549438e-02, -5.22481121e-03, -1.95949401e-02],
                            [5.80919877e-03, 2.57580222e-02, 6.93423314e-03],
                            [3.11775698e-03, 6.69825941e-03, 2.00727748e-03],
                            [2.62073304e-03, 9.94453919e-03, 7.86800192e-03],
                            [-1.36831664e-01, 1.06785026e-04, -5.64310721e-02],
                            [-4.13380289e-02, -1.98769850e-03, -7.29442552e-03],
                            [-3.42883704e-02, -3.90445715e-03, -1.31903698e-02],
                            [-3.98766230e-02, 1.50048575e-02, 4.10491141e-02],
                            [-7.03320227e-03, 3.14323320e-03, 9.82858536e-03],
                            [-3.91029547e-02, -2.50420088e-02, 4.75975138e-02],
                            [-3.19239004e-02, -3.78333796e-02, 1.03070603e-02],
                            [-7.28289756e-03, -1.31227630e-02, 6.04312467e-04],
                            [-2.39672048e-02, -2.77544312e-02, 1.05652401e-02],
                            [-1.69413808e-02, 1.32959670e-02, 4.20292421e-02],
                            [5.80488730e-03, 1.17613126e-02, 1.39260906e-02],
                            [-1.40970191e-02, 8.70572639e-03, 1.27696986e-02],
                            [-2.66591757e-02, -8.20268170e-02, -1.42113498e-02],
                            [1.23204937e-03, -2.97407616e-02, 2.68551812e-03],
                            [-3.37210637e-04, -1.45256899e-02, -3.48008725e-03]])
    assert np.allclose(ref_ind_dip, env.induced_dipoles.induced_dipoles)


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
    mic = False
    box = np.array([])
    for i, field in enumerate(fields):
        starting_guess[i, :] = np.einsum('ij, j', polarizabilities[i], field)
    threshold = 1e-10
    ind_dipoles, iteration = induced_dipoles_jacobi(coordinates, polarizabilities, exclusions, indices,
                                                    fields, starting_guess, mic, box, threshold)
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
                                              fields, starting_guess, mic, box, threshold)
        assert iteration < 500  # arbitrary upper limit for iteration


def test_induced_dipoles_jacobi_edge_cases():
    # Test with minimum input size
    coordinates = np.ones([1, 3], dtype=np.float64)
    polarizabilities = np.ones([1, 3, 3], dtype=np.float64)
    exclusions = [(1,)]
    indices = np.ones(1, dtype=np.int64)
    fields = np.ones([1, 3], dtype=np.float64)
    starting_guess = np.ones([1, 3], dtype=np.float64)
    mic = False
    box = np.array([])
    threshold = 1e-10
    ind_dipoles, _ = induced_dipoles_jacobi(coordinates, polarizabilities, exclusions, indices,
                                            fields, starting_guess, mic, box, threshold)
    assert np.allclose(ind_dipoles, 3 * np.ones([1, 3], dtype=np.float64))


def test_induced_dipoles_jacobi_invalid_inputs():
    # Test with invalid inputs
    with pytest.raises(ValueError, match="Wrong input format."):
        induced_dipoles_jacobi(None,
                               None,
                               None,
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
    exclusions = [(i,) for i in range(3)]
    indices = np.array([0., 1., 2.], dtype=np.int64)
    fields = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float64)
    starting_guess = np.ones([3, 3], dtype=np.float64)
    mic = False
    box = np.array([])
    threshold = 1e-6
    ref_dipoles = np.array([[0.11632816, 0.11632816, 0.11632816],
                            [0.12686485, 0.12686485, 0.12686485],
                            [0.11632816, 0.11632816, 0.11632816]])
    # Call the function multiple times with the same inputs
    for i in range(5):
        ind_dipoles, num_iter = induced_dipoles_jacobi(coordinates, polarizabilities, exclusions, indices,
                                                       fields, starting_guess, mic, box, threshold)
        assert np.allclose(ind_dipoles, ref_dipoles)  # Assert that the output is consistent


def test_induced_dipoles_jacobi_large_inputs():
    # Test number of iterations against tighter thresholds
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json')
    env.solve_induced_dipoles(threshold=1e-8, solver='jacobi')
    assert env.induced_dipoles.number_of_iterations == 16
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json')
    env.solve_induced_dipoles(threshold=1e-10, solver='jacobi')
    assert env.induced_dipoles.number_of_iterations == 20
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json')
    env.solve_induced_dipoles(threshold=1e-15, solver='jacobi')
    assert env.induced_dipoles.number_of_iterations == 30
    # Test error is raised when past the max number of iterations
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json')
    with pytest.raises(RuntimeError, match="Did not converge after the maximum number of iterations."):
        env.solve_induced_dipoles(threshold=1e-1000, solver='jacobi')
    # Test induced dipoles
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json')
    env.solve_induced_dipoles(threshold=1e-8, solver='jacobi')
    ref_ind_dip = np.array([[-5.33083758e-03, 1.80899242e-02, -4.67124873e-02],
                            [-6.25853096e-03, 7.25840460e-03, -1.03916985e-02],
                            [-2.48028626e-02, 2.09508571e-02, -5.74245555e-02],
                            [-4.03295102e-02, -2.81401191e-02, -3.81210336e-02],
                            [-4.06230086e-02, -1.38089059e-02, -4.50891871e-02],
                            [-4.09294307e-05, -7.71654933e-03, -1.10062634e-02],
                            [-1.85295339e-02, 2.86513856e-02, -6.41668069e-03],
                            [-1.89700263e-02, 3.32285459e-02, -1.05088196e-02],
                            [-4.44393262e-03, 6.60779904e-03, -6.06423064e-03],
                            [3.35370357e-02, -1.38589706e-02, -1.33829243e-01],
                            [8.46508800e-03, -1.95121184e-02, -4.13855335e-02],
                            [9.41524338e-02, -7.63125728e-03, -1.05125435e-01],
                            [-3.51664680e-02, -9.46800752e-02, 3.21263242e-02],
                            [-3.05118301e-02, -5.72413535e-02, -7.15883504e-03],
                            [-6.53756795e-03, -1.78294688e-02, 1.00302692e-02],
                            [-5.47042853e-02, -2.59646233e-02, 7.46069721e-02],
                            [-1.21777884e-02, -7.14485835e-03, 1.14801021e-02],
                            [-4.89534262e-03, 8.89451393e-04, 3.05621760e-02],
                            [-2.45470732e-02, 3.50936051e-02, 1.62710897e-02],
                            [-7.08697146e-03, 3.29550988e-02, 2.01553381e-02],
                            [-3.98872500e-02, 2.68723152e-03, 2.96662818e-02],
                            [1.58478302e-01, 1.75607576e-02, -9.62387317e-02],
                            [3.13149664e-02, 1.50598941e-02, 3.26735356e-03],
                            [1.74549438e-02, -5.22481121e-03, -1.95949401e-02],
                            [5.80919877e-03, 2.57580222e-02, 6.93423314e-03],
                            [3.11775698e-03, 6.69825941e-03, 2.00727748e-03],
                            [2.62073304e-03, 9.94453919e-03, 7.86800192e-03],
                            [-1.36831664e-01, 1.06785026e-04, -5.64310721e-02],
                            [-4.13380289e-02, -1.98769850e-03, -7.29442552e-03],
                            [-3.42883704e-02, -3.90445715e-03, -1.31903698e-02],
                            [-3.98766230e-02, 1.50048575e-02, 4.10491141e-02],
                            [-7.03320227e-03, 3.14323320e-03, 9.82858536e-03],
                            [-3.91029547e-02, -2.50420088e-02, 4.75975138e-02],
                            [-3.19239004e-02, -3.78333796e-02, 1.03070603e-02],
                            [-7.28289756e-03, -1.31227630e-02, 6.04312467e-04],
                            [-2.39672048e-02, -2.77544312e-02, 1.05652401e-02],
                            [-1.69413808e-02, 1.32959670e-02, 4.20292421e-02],
                            [5.80488730e-03, 1.17613126e-02, 1.39260906e-02],
                            [-1.40970191e-02, 8.70572639e-03, 1.27696986e-02],
                            [-2.66591757e-02, -8.20268170e-02, -1.42113498e-02],
                            [1.23204937e-03, -2.97407616e-02, 2.68551812e-03],
                            [-3.37210637e-04, -1.45256899e-02, -3.48008725e-03]])
    assert np.allclose(ref_ind_dip, env.induced_dipoles.induced_dipoles)


def test_induced_dipoles_jidiis(
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
    mic = False
    box = np.array([])
    for i, field in enumerate(fields):
        starting_guess[i, :] = np.einsum('ij, j', polarizabilities[i], field)
    threshold = 1e-10
    ind_dipoles, iteration = induced_dipoles_jidiis(coordinates, polarizabilities, exclusions, indices,
                                                    fields, starting_guess, mic, box, threshold)
    # Test if output has the correct dipoles
    assert np.allclose(ind_dipoles, ref_dipoles)
    # Test if the output has the correct shape
    assert ind_dipoles.shape == (6, 3)
    # Test if the iteration count is reasonable
    assert iteration < 500  # arbitrary upper limit for iteration
    # Test for different threshold values
    thresholds = [1e-10, 1e-12, 1e-15]
    for threshold in thresholds:
        _, iteration = induced_dipoles_jidiis(coordinates, polarizabilities, exclusions, indices,
                                              fields, starting_guess, mic, box, threshold)
        assert iteration < 500  # arbitrary upper limit for iteration
    # Test for minimal max_diis
    ind_dipoles, iteration = induced_dipoles_jidiis(coordinates, polarizabilities, exclusions, indices,
                                                    fields, starting_guess, mic, box, threshold, max_diis=1)
    assert np.allclose(ind_dipoles, ref_dipoles)

    # Test DIIS with initialization at the first iteration
    ind_dipoles, iteration = induced_dipoles_dcjidiis(coordinates, polarizabilities, exclusions, indices,
                                                      fields, starting_guess, mic, box, threshold, init_diis=1)
    assert np.allclose(ind_dipoles, ref_dipoles)


def test_induced_dipoles_jidiis_edge_cases():
    # Test with minimum input size
    coordinates = np.ones([1, 3], dtype=np.float64)
    polarizabilities = np.ones([1, 3, 3], dtype=np.float64)
    exclusions = [(1,)]
    indices = np.ones(1, dtype=np.int64)
    fields = np.ones([1, 3], dtype=np.float64)
    starting_guess = np.ones([1, 3], dtype=np.float64)
    mic = False
    box = np.array([])
    threshold = 1e-10
    ind_dipoles, _ = induced_dipoles_jidiis(coordinates, polarizabilities, exclusions, indices,
                                            fields, starting_guess, mic, box, threshold)
    assert np.allclose(ind_dipoles, 3 * np.ones([1, 3], dtype=np.float64))


def test_induced_dipoles_jidiis_invalid_inputs():
    # Test with invalid inputs
    with pytest.raises(ValueError, match="Wrong input format."):
        induced_dipoles_jidiis(None,
                               None,
                               None,
                               None,
                               None,
                               None,
                               None,
                               None,
                               None,
                               None,
                               None,
                               None)


def test_induced_dipoles_jidiis_stability():
    coordinates = np.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]])
    polarizabilities = np.array([[[.1, .1, .1], [.1, .1, .1], [.1, .1, .1]],
                                 [[.1, .1, .1], [.1, .1, .1], [.1, .1, .1]],
                                 [[.1, .1, .1], [.1, .1, .1], [.1, .1, .1]]])
    exclusions = [(i,) for i in range(3)]
    indices = np.array([0., 1., 2.], dtype=np.int64)
    fields = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float64)
    starting_guess = np.ones([3, 3], dtype=np.float64)
    mic = False
    box = np.array([])
    threshold = 1e-6
    ref_dipoles = np.array([[0.11632816, 0.11632816, 0.11632816],
                            [0.12686485, 0.12686485, 0.12686485],
                            [0.11632816, 0.11632816, 0.11632816]])
    # Call the function multiple times with the same inputs
    for i in range(5):
        ind_dipoles, num_iter = induced_dipoles_jidiis(coordinates, polarizabilities, exclusions, indices,
                                                       fields, starting_guess, mic, box, threshold)
        assert np.allclose(ind_dipoles, ref_dipoles)  # Assert that the output is consistent


def test_induced_dipoles_jidiis_large_inputs():
    # Test number of iterations against tighter thresholds
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json')
    env.solve_induced_dipoles(solver='jidiis', threshold=1e-8)
    assert env.induced_dipoles.number_of_iterations <= 16
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json')
    env.solve_induced_dipoles(solver='jidiis', threshold=1e-10)
    assert env.induced_dipoles.number_of_iterations <= 20
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json')
    env.solve_induced_dipoles(solver='jidiis', threshold=1e-15)
    assert env.induced_dipoles.number_of_iterations <= 30
    # Test error is raised when past the max number of iterations
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json')
    with pytest.raises(RuntimeError, match="Did not converge after the maximum number of iterations."):
        env.solve_induced_dipoles(solver='jidiis', threshold=1e-1000)
    # Test induced dipoles
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json')
    env.solve_induced_dipoles(solver='jidiis', threshold=1e-8)
    ref_ind_dip = np.array([[-5.33083758e-03, 1.80899242e-02, -4.67124873e-02],
                            [-6.25853096e-03, 7.25840460e-03, -1.03916985e-02],
                            [-2.48028626e-02, 2.09508571e-02, -5.74245555e-02],
                            [-4.03295102e-02, -2.81401191e-02, -3.81210336e-02],
                            [-4.06230086e-02, -1.38089059e-02, -4.50891871e-02],
                            [-4.09294307e-05, -7.71654933e-03, -1.10062634e-02],
                            [-1.85295339e-02, 2.86513856e-02, -6.41668069e-03],
                            [-1.89700263e-02, 3.32285459e-02, -1.05088196e-02],
                            [-4.44393262e-03, 6.60779904e-03, -6.06423064e-03],
                            [3.35370357e-02, -1.38589706e-02, -1.33829243e-01],
                            [8.46508800e-03, -1.95121184e-02, -4.13855335e-02],
                            [9.41524338e-02, -7.63125728e-03, -1.05125435e-01],
                            [-3.51664680e-02, -9.46800752e-02, 3.21263242e-02],
                            [-3.05118301e-02, -5.72413535e-02, -7.15883504e-03],
                            [-6.53756795e-03, -1.78294688e-02, 1.00302692e-02],
                            [-5.47042853e-02, -2.59646233e-02, 7.46069721e-02],
                            [-1.21777884e-02, -7.14485835e-03, 1.14801021e-02],
                            [-4.89534262e-03, 8.89451393e-04, 3.05621760e-02],
                            [-2.45470732e-02, 3.50936051e-02, 1.62710897e-02],
                            [-7.08697146e-03, 3.29550988e-02, 2.01553381e-02],
                            [-3.98872500e-02, 2.68723152e-03, 2.96662818e-02],
                            [1.58478302e-01, 1.75607576e-02, -9.62387317e-02],
                            [3.13149664e-02, 1.50598941e-02, 3.26735356e-03],
                            [1.74549438e-02, -5.22481121e-03, -1.95949401e-02],
                            [5.80919877e-03, 2.57580222e-02, 6.93423314e-03],
                            [3.11775698e-03, 6.69825941e-03, 2.00727748e-03],
                            [2.62073304e-03, 9.94453919e-03, 7.86800192e-03],
                            [-1.36831664e-01, 1.06785026e-04, -5.64310721e-02],
                            [-4.13380289e-02, -1.98769850e-03, -7.29442552e-03],
                            [-3.42883704e-02, -3.90445715e-03, -1.31903698e-02],
                            [-3.98766230e-02, 1.50048575e-02, 4.10491141e-02],
                            [-7.03320227e-03, 3.14323320e-03, 9.82858536e-03],
                            [-3.91029547e-02, -2.50420088e-02, 4.75975138e-02],
                            [-3.19239004e-02, -3.78333796e-02, 1.03070603e-02],
                            [-7.28289756e-03, -1.31227630e-02, 6.04312467e-04],
                            [-2.39672048e-02, -2.77544312e-02, 1.05652401e-02],
                            [-1.69413808e-02, 1.32959670e-02, 4.20292421e-02],
                            [5.80488730e-03, 1.17613126e-02, 1.39260906e-02],
                            [-1.40970191e-02, 8.70572639e-03, 1.27696986e-02],
                            [-2.66591757e-02, -8.20268170e-02, -1.42113498e-02],
                            [1.23204937e-03, -2.97407616e-02, 2.68551812e-03],
                            [-3.37210637e-04, -1.45256899e-02, -3.48008725e-03]])
    assert np.allclose(ref_ind_dip, env.induced_dipoles.induced_dipoles)


def test_induced_dipoles_dcji(
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
    mic = False
    box = np.array([])
    for i, field in enumerate(fields):
        starting_guess[i, :] = np.einsum('ij, j', polarizabilities[i], field)
    threshold = 1e-10
    ind_dipoles, iteration = induced_dipoles_dcji(coordinates, polarizabilities, exclusions, indices,
                                                  fields, starting_guess, mic, box, threshold)
    # Test if output has the correct dipoles
    assert np.allclose(ind_dipoles, ref_dipoles)
    # Test if the output has the correct shape
    assert ind_dipoles.shape == (6, 3)
    # Test if the iteration count is reasonable
    assert iteration < 500  # arbitrary upper limit for iteration
    # Test for different threshold values
    thresholds = [1e-10, 1e-12, 1e-15]
    for threshold in thresholds:
        _, iteration = induced_dipoles_dcji(coordinates, polarizabilities, exclusions, indices,
                                            fields, starting_guess, mic, box, threshold)
        assert iteration < 500  # arbitrary upper limit for iteration
    # Test clusters with a single atom
    ind_dipoles, iteration = induced_dipoles_dcji(coordinates, polarizabilities, exclusions, indices,
                                                  fields, starting_guess, mic, box, threshold, k_cluster=6)
    assert np.allclose(ind_dipoles, ref_dipoles)
    # Test with all atoms in a single cluster
    with pytest.raises(ValueError,
                       match="k_cluster must be larger than 1 and smaller than or equal to the number of atoms."):
        induced_dipoles_dcji(coordinates, polarizabilities, exclusions, indices,
                             fields, starting_guess, mic, box, threshold, k_cluster=1)


def test_induced_dipoles_dcji_invalid_inputs():
    # Test with invalid inputs
    with pytest.raises(ValueError, match="Wrong input format."):
        induced_dipoles_dcji(None,
                             None,
                             None,
                             None,
                             None,
                             None,
                             None,
                             None,
                             None,
                             None,
                             None,
                             None)


def test_induced_dipoles_dcji_large_inputs():
    # Test number of iterations against tighter thresholds
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json')
    env.solve_induced_dipoles(solver='dcji', threshold=1e-8)
    assert env.induced_dipoles.number_of_iterations <= 16
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json')
    env.solve_induced_dipoles(solver='dcji', threshold=1e-10)
    assert env.induced_dipoles.number_of_iterations <= 20
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json')
    env.solve_induced_dipoles(solver='dcji', threshold=1e-15)
    assert env.induced_dipoles.number_of_iterations <= 30
    # Test error is raised when past the max number of iterations
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json')
    with pytest.raises(RuntimeError, match="Did not converge after the maximum number of iterations."):
        env.solve_induced_dipoles(solver='dcji', threshold=1e-1000)
    # Test induced dipoles
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json')
    env.solve_induced_dipoles(solver='dcji', threshold=1e-8)
    ref_ind_dip = np.array([[-5.33083758e-03, 1.80899242e-02, -4.67124873e-02],
                            [-6.25853096e-03, 7.25840460e-03, -1.03916985e-02],
                            [-2.48028626e-02, 2.09508571e-02, -5.74245555e-02],
                            [-4.03295102e-02, -2.81401191e-02, -3.81210336e-02],
                            [-4.06230086e-02, -1.38089059e-02, -4.50891871e-02],
                            [-4.09294307e-05, -7.71654933e-03, -1.10062634e-02],
                            [-1.85295339e-02, 2.86513856e-02, -6.41668069e-03],
                            [-1.89700263e-02, 3.32285459e-02, -1.05088196e-02],
                            [-4.44393262e-03, 6.60779904e-03, -6.06423064e-03],
                            [3.35370357e-02, -1.38589706e-02, -1.33829243e-01],
                            [8.46508800e-03, -1.95121184e-02, -4.13855335e-02],
                            [9.41524338e-02, -7.63125728e-03, -1.05125435e-01],
                            [-3.51664680e-02, -9.46800752e-02, 3.21263242e-02],
                            [-3.05118301e-02, -5.72413535e-02, -7.15883504e-03],
                            [-6.53756795e-03, -1.78294688e-02, 1.00302692e-02],
                            [-5.47042853e-02, -2.59646233e-02, 7.46069721e-02],
                            [-1.21777884e-02, -7.14485835e-03, 1.14801021e-02],
                            [-4.89534262e-03, 8.89451393e-04, 3.05621760e-02],
                            [-2.45470732e-02, 3.50936051e-02, 1.62710897e-02],
                            [-7.08697146e-03, 3.29550988e-02, 2.01553381e-02],
                            [-3.98872500e-02, 2.68723152e-03, 2.96662818e-02],
                            [1.58478302e-01, 1.75607576e-02, -9.62387317e-02],
                            [3.13149664e-02, 1.50598941e-02, 3.26735356e-03],
                            [1.74549438e-02, -5.22481121e-03, -1.95949401e-02],
                            [5.80919877e-03, 2.57580222e-02, 6.93423314e-03],
                            [3.11775698e-03, 6.69825941e-03, 2.00727748e-03],
                            [2.62073304e-03, 9.94453919e-03, 7.86800192e-03],
                            [-1.36831664e-01, 1.06785026e-04, -5.64310721e-02],
                            [-4.13380289e-02, -1.98769850e-03, -7.29442552e-03],
                            [-3.42883704e-02, -3.90445715e-03, -1.31903698e-02],
                            [-3.98766230e-02, 1.50048575e-02, 4.10491141e-02],
                            [-7.03320227e-03, 3.14323320e-03, 9.82858536e-03],
                            [-3.91029547e-02, -2.50420088e-02, 4.75975138e-02],
                            [-3.19239004e-02, -3.78333796e-02, 1.03070603e-02],
                            [-7.28289756e-03, -1.31227630e-02, 6.04312467e-04],
                            [-2.39672048e-02, -2.77544312e-02, 1.05652401e-02],
                            [-1.69413808e-02, 1.32959670e-02, 4.20292421e-02],
                            [5.80488730e-03, 1.17613126e-02, 1.39260906e-02],
                            [-1.40970191e-02, 8.70572639e-03, 1.27696986e-02],
                            [-2.66591757e-02, -8.20268170e-02, -1.42113498e-02],
                            [1.23204937e-03, -2.97407616e-02, 2.68551812e-03],
                            [-3.37210637e-04, -1.45256899e-02, -3.48008725e-03]])
    assert np.allclose(ref_ind_dip, env.induced_dipoles.induced_dipoles)


def test_induced_dipoles_dcjidiis(
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
    mic = False
    box = np.array([])
    for i, field in enumerate(fields):
        starting_guess[i, :] = np.einsum('ij, j', polarizabilities[i], field)
    threshold = 1e-10
    ind_dipoles, iteration = induced_dipoles_dcjidiis(coordinates, polarizabilities, exclusions, indices,
                                                      fields, starting_guess, mic, box, threshold)
    # Test if output has the correct dipoles
    assert np.allclose(ind_dipoles, ref_dipoles)
    # Test if the output has the correct shape
    assert ind_dipoles.shape == (6, 3)
    # Test if the iteration count is reasonable
    assert iteration < 500  # arbitrary upper limit for iteration
    # Test for different threshold values
    thresholds = [1e-10, 1e-12, 1e-15]
    for threshold in thresholds:
        _, iteration = induced_dipoles_dcjidiis(coordinates, polarizabilities, exclusions, indices,
                                                fields, starting_guess, mic, box, threshold)
        assert iteration < 500  # arbitrary upper limit for iteration
    # Test clusters with a single atom
    ind_dipoles, iteration = induced_dipoles_dcjidiis(coordinates, polarizabilities, exclusions, indices,
                                                      fields, starting_guess, mic, box, threshold, k_cluster=6)
    assert np.allclose(ind_dipoles, ref_dipoles)
    # Test DIIS with initialization at the first iteration
    ind_dipoles, iteration = induced_dipoles_dcjidiis(coordinates, polarizabilities, exclusions, indices,
                                                      fields, starting_guess, mic, box, threshold, init_diis=1)
    assert np.allclose(ind_dipoles, ref_dipoles)
    # Test with all atoms in a single cluster
    with pytest.raises(ValueError,
                       match="k_cluster must be larger than 1 and smaller than or equal to the number of atoms."):
        induced_dipoles_dcjidiis(coordinates, polarizabilities, exclusions, indices,
                                 fields, starting_guess, mic, box, threshold, k_cluster=1)


def test_induced_dipoles_dcjidiis_invalid_inputs():
    # Test with invalid inputs
    with pytest.raises(ValueError, match="Wrong input format."):
        induced_dipoles_dcjidiis(None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None)


def test_induced_dipoles_dcjidiis_large_inputs():
    # Test number of iterations against tighter thresholds
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json')
    env.solve_induced_dipoles(solver='dcjidiis', threshold=1e-8)
    assert env.induced_dipoles.number_of_iterations <= 16
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json')
    env.solve_induced_dipoles(solver='dcjidiis', threshold=1e-10)
    assert env.induced_dipoles.number_of_iterations <= 20
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json')
    env.solve_induced_dipoles(solver='dcjidiis', threshold=1e-15)
    assert env.induced_dipoles.number_of_iterations <= 30
    # Test error is raised when past the max number of iterations
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json')
    with pytest.raises(RuntimeError, match="Did not converge after the maximum number of iterations."):
        env.solve_induced_dipoles(solver='dcjidiis', threshold=1e-1000)
    # Test induced dipoles
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json')
    env.solve_induced_dipoles(solver='dcjidiis', threshold=1e-8)
    ref_ind_dip = np.array([[-5.33083758e-03, 1.80899242e-02, -4.67124873e-02],
                            [-6.25853096e-03, 7.25840460e-03, -1.03916985e-02],
                            [-2.48028626e-02, 2.09508571e-02, -5.74245555e-02],
                            [-4.03295102e-02, -2.81401191e-02, -3.81210336e-02],
                            [-4.06230086e-02, -1.38089059e-02, -4.50891871e-02],
                            [-4.09294307e-05, -7.71654933e-03, -1.10062634e-02],
                            [-1.85295339e-02, 2.86513856e-02, -6.41668069e-03],
                            [-1.89700263e-02, 3.32285459e-02, -1.05088196e-02],
                            [-4.44393262e-03, 6.60779904e-03, -6.06423064e-03],
                            [3.35370357e-02, -1.38589706e-02, -1.33829243e-01],
                            [8.46508800e-03, -1.95121184e-02, -4.13855335e-02],
                            [9.41524338e-02, -7.63125728e-03, -1.05125435e-01],
                            [-3.51664680e-02, -9.46800752e-02, 3.21263242e-02],
                            [-3.05118301e-02, -5.72413535e-02, -7.15883504e-03],
                            [-6.53756795e-03, -1.78294688e-02, 1.00302692e-02],
                            [-5.47042853e-02, -2.59646233e-02, 7.46069721e-02],
                            [-1.21777884e-02, -7.14485835e-03, 1.14801021e-02],
                            [-4.89534262e-03, 8.89451393e-04, 3.05621760e-02],
                            [-2.45470732e-02, 3.50936051e-02, 1.62710897e-02],
                            [-7.08697146e-03, 3.29550988e-02, 2.01553381e-02],
                            [-3.98872500e-02, 2.68723152e-03, 2.96662818e-02],
                            [1.58478302e-01, 1.75607576e-02, -9.62387317e-02],
                            [3.13149664e-02, 1.50598941e-02, 3.26735356e-03],
                            [1.74549438e-02, -5.22481121e-03, -1.95949401e-02],
                            [5.80919877e-03, 2.57580222e-02, 6.93423314e-03],
                            [3.11775698e-03, 6.69825941e-03, 2.00727748e-03],
                            [2.62073304e-03, 9.94453919e-03, 7.86800192e-03],
                            [-1.36831664e-01, 1.06785026e-04, -5.64310721e-02],
                            [-4.13380289e-02, -1.98769850e-03, -7.29442552e-03],
                            [-3.42883704e-02, -3.90445715e-03, -1.31903698e-02],
                            [-3.98766230e-02, 1.50048575e-02, 4.10491141e-02],
                            [-7.03320227e-03, 3.14323320e-03, 9.82858536e-03],
                            [-3.91029547e-02, -2.50420088e-02, 4.75975138e-02],
                            [-3.19239004e-02, -3.78333796e-02, 1.03070603e-02],
                            [-7.28289756e-03, -1.31227630e-02, 6.04312467e-04],
                            [-2.39672048e-02, -2.77544312e-02, 1.05652401e-02],
                            [-1.69413808e-02, 1.32959670e-02, 4.20292421e-02],
                            [5.80488730e-03, 1.17613126e-02, 1.39260906e-02],
                            [-1.40970191e-02, 8.70572639e-03, 1.27696986e-02],
                            [-2.66591757e-02, -8.20268170e-02, -1.42113498e-02],
                            [1.23204937e-03, -2.97407616e-02, 2.68551812e-03],
                            [-3.37210637e-04, -1.45256899e-02, -3.48008725e-03]])
    assert np.allclose(ref_ind_dip, env.induced_dipoles.induced_dipoles)


@pytest.mark.mpi()
def test_induced_dipoles_jacobi_mpi_consistency():
    comm = MPI.COMM_WORLD
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json',
                                  comm=comm)
    env.solve_induced_dipoles(solver='jacobi', threshold=1e-10)
    gathered_summed_arr = comm.gather(env.induced_dipoles.induced_dipoles, root=0)
    gathered_summed_arr2 = comm.gather(env._multipole_fields, root=0)
    gathered_summed_arr3 = comm.gather(env.induced_dipoles.number_of_iterations, root=0)
    # Check if the gathered arrays are equal on all processes
    if comm.Get_rank() == 0:
        for i in range(1, comm.Get_size()):
            assert np.array_equal(gathered_summed_arr[0],
                                  gathered_summed_arr[i]), "Arrays are not equal across processes"
            assert np.array_equal(gathered_summed_arr2[0],
                                  gathered_summed_arr2[i]), "Arrays are not equal across processes"
            assert np.array_equal(gathered_summed_arr3[0],
                                  gathered_summed_arr3[i]), "Arrays are not equal across processes"
    # MPI.Finalize()


@pytest.mark.mpi()
def test_induced_dipoles_jidiis_mpi_consistency():
    comm = MPI.COMM_WORLD
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json',
                                  comm=comm)
    env.solve_induced_dipoles(solver='jidiis', threshold=1e-10)
    gathered_summed_arr = comm.gather(env.induced_dipoles.induced_dipoles, root=0)
    gathered_summed_arr2 = comm.gather(env._multipole_fields, root=0)
    gathered_summed_arr3 = comm.gather(env.induced_dipoles.number_of_iterations, root=0)
    # Check if the gathered arrays are equal on all processes
    if comm.Get_rank() == 0:
        for i in range(1, comm.Get_size()):
            assert np.array_equal(gathered_summed_arr[0],
                                  gathered_summed_arr[i]), "Arrays are not equal across processes"
            assert np.array_equal(gathered_summed_arr2[0],
                                  gathered_summed_arr2[i]), "Arrays are not equal across processes"
            assert np.array_equal(gathered_summed_arr3[0],
                                  gathered_summed_arr3[i]), "Arrays are not equal across processes"
    # MPI.Finalize()


@pytest.mark.mpi()
def test_induced_dipoles_dcji_mpi_consistency():
    comm = MPI.COMM_WORLD
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json',
                                  comm=comm)
    env.solve_induced_dipoles(solver='dcji', threshold=1e-10)
    gathered_summed_arr = comm.gather(env.induced_dipoles.induced_dipoles, root=0)
    gathered_summed_arr2 = comm.gather(env._multipole_fields, root=0)
    gathered_summed_arr3 = comm.gather(env.induced_dipoles.number_of_iterations, root=0)
    # Check if the gathered arrays are equal on all processes
    if comm.Get_rank() == 0:
        for i in range(1, comm.Get_size()):
            assert np.array_equal(gathered_summed_arr[0],
                                  gathered_summed_arr[i]), "Arrays are not equal across processes"
            assert np.array_equal(gathered_summed_arr2[0],
                                  gathered_summed_arr2[i]), "Arrays are not equal across processes"
            assert np.array_equal(gathered_summed_arr3[0],
                                  gathered_summed_arr3[i]), "Arrays are not equal across processes"
    # MPI.Finalize()


@pytest.mark.mpi()
def test_induced_dipoles_dcjidiis_mpi_consistency():
    comm = MPI.COMM_WORLD
    core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json',
                                  comm=comm)
    env.solve_induced_dipoles(solver='dcjidiis', threshold=1e-10)
    gathered_summed_arr = comm.gather(env.induced_dipoles.induced_dipoles, root=0)
    gathered_summed_arr2 = comm.gather(env._multipole_fields, root=0)
    gathered_summed_arr3 = comm.gather(env.induced_dipoles.number_of_iterations, root=0)
    # Check if the gathered arrays are equal on all processes
    if comm.Get_rank() == 0:
        for i in range(1, comm.Get_size()):
            assert np.array_equal(gathered_summed_arr[0],
                                  gathered_summed_arr[i]), "Arrays are not equal across processes"
            assert np.array_equal(gathered_summed_arr2[0],
                                  gathered_summed_arr2[i]), "Arrays are not equal across processes"
            assert np.array_equal(gathered_summed_arr3[0],
                                  gathered_summed_arr3[i]), "Arrays are not equal across processes"
    MPI.Finalize()


def test_diis_stability():
    # test ind_dipoles matrix of 2 dimensions instead of 3 and singluar matrix
    ind_dipoles = np.array([[-0.05130705, -0.25165223, -0.4511121],
                            [0.00465468, -0.27383291, -0.18815435],
                            [-0.09539097, 0.03976932, -0.19509574],
                            [-1.17424447, 0.51284179, 0.50156209],
                            [-0.14716566, -0.03273562, 0.12889495],
                            [-0.44555632, 0.00482074, -0.09598545]])
    max_diis = 10
    iteration = 5
    error_matrix_full = np.ones([3, 3])
    with pytest.raises(ValueError, match="Singular Matrix."):
        direct_inversion_iterative_subspace(ind_dipoles=ind_dipoles,
                                            error_matrix_full=error_matrix_full,
                                            max_diis=max_diis,
                                            iteration=iteration)


def test_kmeans_stability():
    coordinates = np.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]])
    with pytest.raises(ValueError,
                       match="k_cluster must be larger than 1 and smaller than or equal to the number of atoms."):
        kmeans_clustering(coordinates=coordinates, k_cluster=1, cluster_size_range=0)


def test_divide_and_conquer_stability():
    coordinates = np.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]])
    polarizabilities = np.array([[[.1, .1, .1], [.1, .1, .1], [.1, .1, .1]],
                                 [[.1, .1, .1], [.1, .1, .1], [.1, .1, .1]],
                                 [[.1, .1, .1], [.1, .1, .1], [.1, .1, .1]]])
    exclusions = [(i,) for i in range(3)]
    indices = np.array([0, 1, 2])
    mic = False
    clusters = np.array([0, 0, 1])
    with pytest.raises(ValueError, match="Singular Matrix."):
        divide_and_conquer(coordinates, polarizabilities, clusters, exclusions, indices, mic, k_cluster=2)
