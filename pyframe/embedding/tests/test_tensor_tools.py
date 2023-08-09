"""Tests PyFraME.embedding.tensor_tools.py"""
import copy
import numpy as np
import math
from pyframe.embedding import tensor_tools


def test_compute_factorials():
    ref_factorials = np.array([1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0, 3628800.0, 39916800.0,
                               479001600.0])
    factorials = tensor_tools.compute_factorials(12)
    for i in range(len(ref_factorials)):
        assert ref_factorials[i] == factorials[i]


def test_compute_double_factorials():
    ref_double_factorials = np.array([1.0, 1.0, 2.0, 3.0, 8.0, 15.0, 48.0, 105.0, 384.0, 945.0, 3840.0, 10395.0,
                                      46080.0])
    double_factorials = tensor_tools.compute_double_factorials(12)
    for i in range(len(ref_double_factorials)):
        assert ref_double_factorials[i] == double_factorials[i]


def test_compute_binomial_coefficients():
    ref_binomials = np.zeros((13, 13))
    ref_binomials[:, 0] = 1.0
    for i in range(13):
        ref_binomials[i, i] = 1.0
    binomials = tensor_tools.compute_binomial_coefficients(13, 13)
    for i in range(1, 13):
        for j in range(1, i):
            ref_binomials[i, j] = ref_binomials[i - 1, j - 1] + ref_binomials[i - 1, j]
            assert np.allclose(ref_binomials[i, j], binomials[i, j])


def test_compute_trinomial_coefficients():
    binomial_coefficients = tensor_tools.compute_binomial_coefficients(42, 14)
    trinomial_coefficients = tensor_tools.compute_trinomial_coefficients(14, 14, 14, binomial_coefficients)
    for i in range(13):
        for j in range(13):
            for k in range(13):
                if i + j + k > 12:
                    continue
                ref_trinomial = math.factorial(i + j + k) / (math.factorial(i) * math.factorial(j) * math.factorial(k))
                assert math.isclose(ref_trinomial, trinomial_coefficients[i, j, k])


def test_compute_degeneracy_tensor():
    binomial_coefficients = tensor_tools.compute_binomial_coefficients(42, 14)
    trinomial_coefficients = tensor_tools.compute_trinomial_coefficients(14, 14, 14, binomial_coefficients)
    # 0th order
    g_ref = [1.0]
    assert np.allclose(tensor_tools.compute_degeneracy_tensor(tensor_rank=0,
                                                              trinomial_coefficients=trinomial_coefficients), g_ref)
    # 1st order
    g_ref = [1.0, 1.0, 1.0]
    assert np.allclose(tensor_tools.compute_degeneracy_tensor(tensor_rank=1,
                                                              trinomial_coefficients=trinomial_coefficients), g_ref)
    # 2nd order
    g_ref = [1.0, 2.0, 2.0, 1.0, 2.0, 1.0]
    assert np.allclose(tensor_tools.compute_degeneracy_tensor(tensor_rank=2,
                                                              trinomial_coefficients=trinomial_coefficients), g_ref)
    # 3rd order
    g_ref = [1.0, 3.0, 3.0, 3.0, 6.0, 3.0, 1.0, 3.0, 3.0, 1.0]
    assert np.allclose(tensor_tools.compute_degeneracy_tensor(tensor_rank=3,
                                                              trinomial_coefficients=trinomial_coefficients), g_ref)
    # 4th order
    g_ref = [1.0, 4.0, 4.0, 6.0, 12.0, 6.0, 4.0, 12.0, 12.0, 4.0, 1.0, 4.0, 6.0, 4.0, 1.0]
    assert np.allclose(tensor_tools.compute_degeneracy_tensor(tensor_rank=4,
                                                              trinomial_coefficients=trinomial_coefficients), g_ref)
    # 5th order
    g_ref = [1.0, 5.0, 5.0, 10.0, 20.0, 10.0, 10.0, 30.0, 30.0, 10.0, 5.0, 20.0, 30.0, 20.0, 5.0, 1.0, 5.0, 10.0, 10.0,
             5.0, 1.0]
    assert np.allclose(tensor_tools.compute_degeneracy_tensor(tensor_rank=5,
                                                              trinomial_coefficients=trinomial_coefficients), g_ref)


def test_get_tensor_rank():
    # length 0
    tensor = np.zeros(0)
    assert -1 == tensor_tools.rank(tensor)
    # length 1
    tensor = np.zeros(1)
    assert 0 == tensor_tools.rank(tensor)
    # length 3
    tensor = np.zeros(3)
    assert 1 == tensor_tools.rank(tensor)
    # length 6
    tensor = np.zeros(6)
    assert 2 == tensor_tools.rank(tensor)
    # length 10
    tensor = np.zeros(10)
    assert 3 == tensor_tools.rank(tensor)
    # length 15
    tensor = np.zeros(15)
    assert 4 == tensor_tools.rank(tensor)
    # length 136
    tensor = np.zeros(136)
    assert 15 == tensor_tools.rank(tensor)
    # length 136 sliced to 6
    tensor = np.zeros(136)
    tensor = tensor[6:12]
    assert 2 == tensor_tools.rank(tensor)


def test_get_tensor_length():
    # 0th rank
    assert 1 == tensor_tools.length(tensor_rank=0)
    # 1st rank
    assert 3 == tensor_tools.length(tensor_rank=1)
    # 2nd rank
    assert 6 == tensor_tools.length(tensor_rank=2)
    # 3rd rank
    assert 10 == tensor_tools.length(tensor_rank=3)
    # 4th rank
    assert 15 == tensor_tools.length(tensor_rank=4)
    # 5th rank
    assert 21 == tensor_tools.length(tensor_rank=5)
    # 15th rank
    assert 136 == tensor_tools.length(tensor_rank=15)


def test_convert_multi_index():
    for rank in range(13):
        ref_tensor_idx = 1
        for i in range(rank, -1, -1):
            for j in range(rank - i, -1, -1):
                k = rank - i - j
                assert ref_tensor_idx == tensor_tools.convert_multi_index((i, j, k))
                ref_tensor_idx += 1


def test_convert_tensor_index():
    for rank in range(13):
        idx = 1
        for i in range(rank, -1, -1):
            for j in range(rank - i, -1, -1):
                k = rank - i - j
                ref_multiindex = np.array([i, j, k])
                assert np.allclose(ref_multiindex, tensor_tools.convert_tensor_index(idx, rank))
                idx += 1


def test_compute_tensor_coefficients():
    max_order = 3
    tensor_coefficients = tensor_tools.compute_tensor_coefficients(max_order)
    expected_shape = (max_order + 1, max_order + 1, 2 * max_order + 2)
    assert tensor_coefficients.shape == expected_shape

    # Test some specific values of the tensor coefficients
    assert np.allclose(tensor_coefficients[0, 0, 0], 1.0)
    assert np.allclose(tensor_coefficients[2, 2, 3], 15.0)
    assert np.allclose(tensor_coefficients[1, 3, 7], 189.0)


def test_compute_interaction_tensor_element():
    tensor_coefficient = tensor_tools.compute_tensor_coefficients(16)
    # first test
    r_i = np.array([1.0, 2.3, -9.3])
    r_j = np.array([-7.1, -4.5, -20.2])
    r_ij = r_j - r_i
    r_ji = r_i - r_j
    # 0th order derivatives
    multi_indices_2 = [np.array([0, 0, 0]), np.array([0, 0, 0])]
    assert math.isclose(0.0658436437005385, tensor_tools.compute_interaction_tensor_element(multi_indices_2, r_ij,
                                                                                            tensor_coefficient))
    assert math.isclose(0.0658436437005385, tensor_tools.compute_interaction_tensor_element(multi_indices_2, r_ji,
                                                                                            tensor_coefficient))
    # 1st order derivatives
    multi_indices_2 = [np.array([1, 0, 0]), np.array([0, 0, 0])]
    multi_indices_4 = [np.array([0, 1, 0]), np.array([0, 0, 0])]
    multi_indices_5 = [np.array([0, 0, 1]), np.array([0, 0, 0])]
    assert math.isclose(0.0023122063382223267, tensor_tools.compute_interaction_tensor_element(multi_indices_2, r_ij,
                                                                                               tensor_coefficient))
    assert math.isclose(-0.0023122063382223267, tensor_tools.compute_interaction_tensor_element(multi_indices_2, r_ji,
                                                                                                tensor_coefficient))
    assert math.isclose(0.0019411114938162743, tensor_tools.compute_interaction_tensor_element(multi_indices_4, r_ij,
                                                                                               tensor_coefficient))
    assert math.isclose(-0.0019411114938162743, tensor_tools.compute_interaction_tensor_element(multi_indices_4, r_ji,
                                                                                                tensor_coefficient))
    assert math.isclose(0.0031114875415584396, tensor_tools.compute_interaction_tensor_element(multi_indices_5, r_ij,
                                                                                               tensor_coefficient))
    assert math.isclose(-0.0031114875415584396, tensor_tools.compute_interaction_tensor_element(multi_indices_5, r_ji,
                                                                                                tensor_coefficient))
    # 2nd order derivatives
    multi_indices_1 = [np.array([2, 0, 0]), np.array([0, 0, 0])]
    multi_indices_2 = [np.array([1, 0, 0]), np.array([1, 0, 0])]
    multi_indices_3 = [np.array([0, 0, 0]), np.array([2, 0, 0])]
    multi_indices_4 = [np.array([1, 0, 1]), np.array([0, 0, 0])]
    multi_indices_5 = [np.array([0, 1, 1]), np.array([0, 0, 0])]
    multi_indices_6 = [np.array([0, 0, 2]), np.array([0, 0, 0])]
    assert math.isclose(-4.1866945641792955e-5, tensor_tools.compute_interaction_tensor_element(multi_indices_1, r_ij,
                                                                                                tensor_coefficient))
    assert math.isclose(-4.1866945641792955e-5, tensor_tools.compute_interaction_tensor_element(multi_indices_1, r_ji,
                                                                                                tensor_coefficient))
    assert math.isclose(4.1866945641792955e-5, tensor_tools.compute_interaction_tensor_element(multi_indices_2, r_ij,
                                                                                               tensor_coefficient))
    assert math.isclose(4.1866945641792955e-5, tensor_tools.compute_interaction_tensor_element(multi_indices_2, r_ji,
                                                                                               tensor_coefficient))
    assert math.isclose(-4.1866945641792955e-5, tensor_tools.compute_interaction_tensor_element(multi_indices_3, r_ij,
                                                                                                tensor_coefficient))
    assert math.isclose(-4.1866945641792955e-5, tensor_tools.compute_interaction_tensor_element(multi_indices_3, r_ji,
                                                                                                tensor_coefficient))
    assert math.isclose(0.00032779479432875263, tensor_tools.compute_interaction_tensor_element(multi_indices_4, r_ij,
                                                                                                tensor_coefficient))
    assert math.isclose(0.00032779479432875263, tensor_tools.compute_interaction_tensor_element(multi_indices_4, r_ji,
                                                                                                tensor_coefficient))
    assert math.isclose(0.0002751857532636442, tensor_tools.compute_interaction_tensor_element(multi_indices_5, r_ij,
                                                                                               tensor_coefficient))
    assert math.isclose(0.0002751857532636442, tensor_tools.compute_interaction_tensor_element(multi_indices_5, r_ji,
                                                                                               tensor_coefficient))
    assert math.isclose(0.00015564900246433047, tensor_tools.compute_interaction_tensor_element(multi_indices_6, r_ij,
                                                                                                tensor_coefficient))
    assert math.isclose(0.00015564900246433047, tensor_tools.compute_interaction_tensor_element(multi_indices_6, r_ji,
                                                                                                tensor_coefficient))
    # 3rd order derivatives
    multi_indices_1 = [np.array([0, 0, 0]), np.array([0, 3, 0])]
    multi_indices_2 = [np.array([0, 3, 0]), np.array([0, 0, 0])]
    multi_indices_3 = [np.array([0, 0, 0]), np.array([0, 1, 2])]
    multi_indices_4 = [np.array([1, 1, 1]), np.array([0, 0, 0])]
    multi_indices_5 = [np.array([0, 1, 2]), np.array([0, 0, 0])]
    multi_indices_6 = [np.array([0, 1, 0]), np.array([0, 0, 2])]
    assert math.isclose(5.0433694213263545e-5, tensor_tools.compute_interaction_tensor_element(multi_indices_1, r_ij,
                                                                                               tensor_coefficient))
    assert math.isclose(-5.0433694213263545e-5, tensor_tools.compute_interaction_tensor_element(multi_indices_1, r_ji,
                                                                                                tensor_coefficient))
    assert math.isclose(-5.0433694213263545e-5, tensor_tools.compute_interaction_tensor_element(multi_indices_2, r_ij,
                                                                                                tensor_coefficient))
    assert math.isclose(5.0433694213263545e-5, tensor_tools.compute_interaction_tensor_element(multi_indices_2, r_ji,
                                                                                               tensor_coefficient))
    assert math.isclose(4.831797020366597e-5, tensor_tools.compute_interaction_tensor_element(multi_indices_4, r_ij,
                                                                                              tensor_coefficient))
    assert math.isclose(-4.831797020366597e-5, tensor_tools.compute_interaction_tensor_element(multi_indices_4, r_ji,
                                                                                               tensor_coefficient))
    assert math.isclose(-3.977407904023146e-5, tensor_tools.compute_interaction_tensor_element(multi_indices_3, r_ij,
                                                                                              tensor_coefficient))
    assert math.isclose(3.977407904023146e-5, tensor_tools.compute_interaction_tensor_element(multi_indices_3, r_ji,
                                                                                               tensor_coefficient))
    assert math.isclose(3.977407904023146e-5, tensor_tools.compute_interaction_tensor_element(multi_indices_5, r_ij,
                                                                                              tensor_coefficient))
    assert math.isclose(-3.977407904023146e-5, tensor_tools.compute_interaction_tensor_element(multi_indices_5, r_ji,
                                                                                               tensor_coefficient))
    assert math.isclose(3.977407904023146e-5, tensor_tools.compute_interaction_tensor_element(multi_indices_6, r_ij,
                                                                                              tensor_coefficient))
    assert math.isclose(-3.977407904023146e-5, tensor_tools.compute_interaction_tensor_element(multi_indices_6, r_ji,
                                                                                               tensor_coefficient))
    # 4th order derivatives
    multi_indices_2 = [np.array([2, 2, 0]), np.array([0, 0, 0])]
    multi_indices_4 = [np.array([1, 2, 1]), np.array([0, 0, 0])]
    multi_indices_5 = [np.array([0, 0, 4]), np.array([0, 0, 0])]
    assert math.isclose(2.1207330998015187e-6, tensor_tools.compute_interaction_tensor_element(multi_indices_2, r_ij,
                                                                                               tensor_coefficient))
    assert math.isclose(2.1207330998015187e-6, tensor_tools.compute_interaction_tensor_element(multi_indices_2, r_ji,
                                                                                               tensor_coefficient))
    assert math.isclose(2.8655224575164097e-6, tensor_tools.compute_interaction_tensor_element(multi_indices_4, r_ij,
                                                                                               tensor_coefficient))
    assert math.isclose(2.8655224575164097e-6, tensor_tools.compute_interaction_tensor_element(multi_indices_4, r_ji,
                                                                                               tensor_coefficient))
    assert math.isclose(-1.1756644987695054e-5, tensor_tools.compute_interaction_tensor_element(multi_indices_5, r_ij,
                                                                                                tensor_coefficient))
    assert math.isclose(-1.1756644987695054e-5, tensor_tools.compute_interaction_tensor_element(multi_indices_5, r_ji,
                                                                                                tensor_coefficient))
    # 5th order derivatives
    multi_indices_2 = [np.array([2, 1, 2]), np.array([0, 0, 0])]
    multi_indices_4 = [np.array([1, 1, 3]), np.array([0, 0, 0])]
    assert math.isclose(2.5358289633406297e-6, tensor_tools.compute_interaction_tensor_element(multi_indices_2, r_ij,
                                                                                               tensor_coefficient))
    assert math.isclose(-2.5358289633406297e-6, tensor_tools.compute_interaction_tensor_element(multi_indices_2, r_ji,
                                                                                                tensor_coefficient))
    assert math.isclose(2.3986145397364605e-6, tensor_tools.compute_interaction_tensor_element(multi_indices_4, r_ij,
                                                                                               tensor_coefficient))
    assert math.isclose(-2.3986145397364605e-6, tensor_tools.compute_interaction_tensor_element(multi_indices_4, r_ji,
                                                                                                tensor_coefficient))
    # 10th order derivatives
    multi_indices_2 = [np.array([5, 2, 3]), np.array([0, 0, 0])]
    multi_indices_4 = [np.array([3, 6, 1]), np.array([0, 0, 0])]
    assert math.isclose(-2.4842594513699081e-8, tensor_tools.compute_interaction_tensor_element(multi_indices_2, r_ij,
                                                                                                tensor_coefficient))
    assert math.isclose(-2.4842594513699081e-8, tensor_tools.compute_interaction_tensor_element(multi_indices_2, r_ji,
                                                                                                tensor_coefficient))
    assert math.isclose(-5.4055195481204153e-9, tensor_tools.compute_interaction_tensor_element(multi_indices_4, r_ij,
                                                                                                tensor_coefficient))
    assert math.isclose(-5.4055195481204153e-9, tensor_tools.compute_interaction_tensor_element(multi_indices_4, r_ji,
                                                                                                tensor_coefficient))
    # 21st order derivatives
    multi_indices_2 = [np.array([13, 3, 5]), np.array([0, 0, 0])]
    multi_indices_4 = [np.array([7, 7, 7]), np.array([0, 0, 0])]
    assert math.isclose(-7.5730234910186237e-9, tensor_tools.compute_interaction_tensor_element(multi_indices_2, r_ij,
                                                                                                tensor_coefficient))
    assert math.isclose(7.5730234910186237e-9, tensor_tools.compute_interaction_tensor_element(multi_indices_2, r_ji,
                                                                                               tensor_coefficient))
    assert math.isclose(9.8765395046408577e-10, tensor_tools.compute_interaction_tensor_element(multi_indices_4, r_ij,
                                                                                                tensor_coefficient))
    assert math.isclose(-9.8765395046408577e-10, tensor_tools.compute_interaction_tensor_element(multi_indices_4, r_ji,
                                                                                                 tensor_coefficient))
    # second test
    r = np.array([2, 0, 0])
    multi_indices_2 = [np.array([0, 0, 0]), np.array([0, 0, 0])]
    multi_indices_4 = [np.array([1, 0, 0]), np.array([0, 0, 0])]
    multi_indices_5 = [np.array([0, 1, 0]), np.array([0, 0, 0])]
    multi_indices_6 = [np.array([3, 0, 0]), np.array([0, 0, 0])]

    null_derivative_element = tensor_tools.compute_interaction_tensor_element(multi_indices_2, r, tensor_coefficient)
    assert null_derivative_element == 0.5

    first_derivative_x_element = tensor_tools.compute_interaction_tensor_element(multi_indices_4, r, tensor_coefficient)

    assert first_derivative_x_element == -0.25

    first_derivative_y_element = tensor_tools.compute_interaction_tensor_element(multi_indices_5, r, tensor_coefficient)
    assert first_derivative_y_element == 0.0

    third_derivative_x_element = tensor_tools.compute_interaction_tensor_element(multi_indices_6, r, tensor_coefficient)
    assert third_derivative_x_element == -0.375


def test_compute_trace():
    binomial_coefficients = tensor_tools.compute_binomial_coefficients(6, 2)
    trinomial_coefficients = tensor_tools.compute_trinomial_coefficients(2, 2, 2, binomial_coefficients)
    quadrupole = np.array([-3.9516312016, -0.0561791973, 0.0008348984, -4.5778807726, 0.0000430036, -5.0206878337])
    octopole = np.array([-0.0150265431, -0.0987395310, -0.0000662324, 0.0267925411, -0.0002404801, 0.0219905017,
                         0.0316173819, -0.0000537894, 0.2063402594, 0.0001754735])
    # trace quadrupole
    ref_trace = quadrupole[0] + quadrupole[3] + quadrupole[5]
    assert np.isclose(ref_trace, tensor_tools.compute_trace(quadrupole, (0, 0, 0), trinomial_coefficients))
    # traces octopole
    ref_trace = octopole[0] + octopole[3] + octopole[5]
    assert np.isclose(ref_trace, tensor_tools.compute_trace(octopole, (1, 0, 0), trinomial_coefficients))
    ref_trace = octopole[1] + octopole[6] + octopole[8]
    assert np.isclose(ref_trace, tensor_tools.compute_trace(octopole, (0, 1, 0), trinomial_coefficients))
    ref_trace = octopole[2] + octopole[7] + octopole[9]
    assert np.isclose(ref_trace, tensor_tools.compute_trace(octopole, (0, 0, 1), trinomial_coefficients))


def test_detrace():
    factorial = tensor_tools.compute_factorials(3)
    double_factorial = tensor_tools.compute_double_factorials(5)
    binomial_coefficients = tensor_tools.compute_binomial_coefficients(3, 1)
    trinomial_coefficients = tensor_tools.compute_trinomial_coefficients(1, 1, 1, binomial_coefficients)
    quadrupole = np.array([-3.9516312016, -0.0561791973, 0.0008348984, -4.5778807726, 0.0000430036, -5.0206878337])
    octopole = np.array([-0.0150265431, -0.0987395310, -0.0000662324, 0.0267925411, -0.0002404801, 0.0219905017,
                         0.0316173819, -0.0000537894, 0.2063402594, 0.0001754735])
    traceless_quadrupole = copy.deepcopy(quadrupole)
    traceless_octopole = copy.deepcopy(octopole)
    # Calculate trace and traceless quadrupole and octopole
    trace_quadrupole = np.sum([quadrupole[0], quadrupole[3], quadrupole[5]]) / 3.0
    traceless_quadrupole[[0, 3, 5]] -= trace_quadrupole
    trace_octopole = (octopole[0] + octopole[3] + octopole[5]) / 5.0
    traceless_octopole[0] -= 3.0 * trace_octopole
    traceless_octopole[3] -= trace_octopole
    traceless_octopole[5] -= trace_octopole
    trace_octopole = (octopole[1] + octopole[6] + octopole[8]) / 5.0
    traceless_octopole[1] -= trace_octopole
    traceless_octopole[6] -= 3.0 * trace_octopole
    traceless_octopole[8] -= trace_octopole
    trace_octopole = (octopole[2] + octopole[7] + octopole[9]) / 5.0
    traceless_octopole[2] -= trace_octopole
    traceless_octopole[7] -= trace_octopole
    traceless_octopole[9] -= 3.0 * trace_octopole
    # detrace quadrupole and octopole
    detraced_quadrupole = tensor_tools.detrace(quadrupole, factorial, double_factorial, trinomial_coefficients)
    assert np.allclose(detraced_quadrupole, traceless_quadrupole)
    detraced_octopole = tensor_tools.detrace(octopole, factorial, double_factorial, trinomial_coefficients)
    assert np.allclose(detraced_octopole, traceless_octopole)
    # detrace quadrupole and octopole again, to make sure it is not changed
    detraced_quadrupole_2 = tensor_tools.detrace(detraced_quadrupole, factorial, double_factorial,
                                                 trinomial_coefficients)
    assert np.allclose(detraced_quadrupole_2, detraced_quadrupole)
    detraced_octopole_2 = tensor_tools.detrace(detraced_octopole, factorial, double_factorial,
                                               trinomial_coefficients)
    assert np.allclose(detraced_octopole_2, detraced_octopole)
