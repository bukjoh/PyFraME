"""Tests PyFraME.embedding.polytensor.py"""
import pytest
import numpy as np

from pyframe.embedding import polytensor


# Test FirstDegreePolytensor
def test_init_first_degree():
    polytensor_1_obj = polytensor.FirstDegreePolytensor(rank=5)
    assert polytensor_1_obj.length == 56
    polytensor_2_obj = polytensor.FirstDegreePolytensor(rank=3)
    assert polytensor_2_obj.length == 20
    polytensor_3_obj = polytensor.FirstDegreePolytensor(rank=0)
    assert polytensor_3_obj.length == 1
    tensor_data = np.arange(56)
    polytensor_4_obj = polytensor.FirstDegreePolytensor(rank=7, tensor_data=tensor_data)
    assert np.allclose(tensor_data, polytensor_4_obj.data)


def test_write_to_data_block_wise():
    polytensor_1_obj = polytensor.FirstDegreePolytensor(rank=2)
    compressed_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    polytensor_1_obj.write_to_data_block_wise(compressed_tensor)
    assert np.allclose(polytensor_1_obj.data[4:], compressed_tensor)
    polytensor_2_obj = polytensor.FirstDegreePolytensor(rank=0)
    compressed_tensor = np.array([1.0])
    polytensor_2_obj.write_to_data_block_wise(compressed_tensor)
    assert compressed_tensor == polytensor_2_obj.data
    polytensor_3_obj = polytensor.FirstDegreePolytensor(rank=5)
    compressed_tensor = np.arange(21)
    polytensor_3_obj.write_to_data_block_wise(compressed_tensor)
    assert np.allclose(polytensor_3_obj.data[35:], compressed_tensor)
    polytensor_4_obj = polytensor.FirstDegreePolytensor(rank=5)
    compressed_tensor = np.arange(10)
    polytensor_4_obj.write_to_data_block_wise(compressed_tensor)
    assert np.allclose(polytensor_4_obj.data[10:20], compressed_tensor)


def test_write_to_data_first_degree():
    polytensor_1_obj = polytensor.FirstDegreePolytensor(rank=4)
    y = np.zeros(6, dtype=int)
    rank_2_compressed_tensor = np.full_like(y, 1)
    polytensor_1_obj.write_to_data_block_wise(rank_2_compressed_tensor)
    assert_array_1 = np.array([0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for i in range(len(assert_array_1)):
        assert polytensor_1_obj.data[i] == assert_array_1[i]

    z = np.zeros(1, dtype=int)
    rank_0_compressed_tensor = np.full_like(z, 1)
    polytensor_1_obj.write_to_data_block_wise(rank_0_compressed_tensor)
    assert_array_2 = np.array([1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for i in range(len(assert_array_2)):
        assert polytensor_1_obj.data[i] == assert_array_2[i]


def test_add():
    polytensor_1_obj = polytensor.FirstDegreePolytensor(rank=2, data_type=int)
    y = np.zeros(6, dtype=int)
    rank_2_compressed_tensor = np.full_like(y, 1)
    polytensor_1_obj.write_to_data_block_wise(rank_2_compressed_tensor)

    polytensor_2_obj = polytensor.FirstDegreePolytensor(rank=2, data_type=int)
    z = np.zeros(3, dtype=int)
    rank_1_compressed_tensor = np.full_like(z, 1)
    polytensor_2_obj.write_to_data_block_wise(rank_1_compressed_tensor)
    polytensor_3_obj = polytensor_1_obj.__add__(polytensor_2_obj)
    assert_array_1 = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    for i in range(len(assert_array_1)):
        assert polytensor_3_obj.data[i] == assert_array_1[i]


def test_multiply_elementwise():
    a = np.arange(4)
    polytensor_1_obj = polytensor.FirstDegreePolytensor(rank=1, tensor_data=a)
    b = np.arange(4)
    b = np.full_like(b, 2)
    polytensor_2_obj = polytensor.FirstDegreePolytensor(rank=1, tensor_data=b)
    assert np.array_equal(np.multiply(a, b), polytensor_1_obj.multiply_elementwise(polytensor_2_obj).data)
    c = np.arange(56)
    polytensor_1_obj = polytensor.FirstDegreePolytensor(rank=5, tensor_data=c)
    d = np.arange(56)
    d = np.full_like(d, 2)
    polytensor_2_obj = polytensor.FirstDegreePolytensor(rank=5, tensor_data=d)
    assert np.array_equal(np.multiply(c, d), polytensor_1_obj.multiply_elementwise(polytensor_2_obj).data)


def test_multiply_scalar_matrix():
    scalar_data = np.array([1, 2, 3, 4])
    matrices_data = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]])
    polytensor_1_obj = polytensor.FirstDegreePolytensor(rank=1, tensor_data=scalar_data)
    polytensor_2_obj = polytensor.FirstDegreePolytensor(rank=1, tensor_data=matrices_data)
    assert_array_1 = [[10, 10], [10, 10]]
    for i in range(len(assert_array_1)):
        for j in range(len(assert_array_1)):
            assert assert_array_1[i][j] == polytensor_1_obj.multiply_scalar_matrix(polytensor_2_obj)[i][j]
    scalar_data = np.arange(56)
    matrices_data = np.zeros([56, 2, 2])
    matrices_data = np.full_like(matrices_data, 1)
    polytensor_1_obj = polytensor.FirstDegreePolytensor(rank=5, tensor_data=scalar_data)
    polytensor_2_obj = polytensor.FirstDegreePolytensor(rank=5, tensor_data=matrices_data)
    result = np.zeros([2, 2])
    for i in range(56):
        result += scalar_data[i] * matrices_data[i]
    assert np.allclose(result, polytensor_1_obj.multiply_scalar_matrix(polytensor_2_obj))


def test_multiply_matrix_scalar():
    scalar_data = np.array([1, 2, 3, 4])
    matrices_data = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]])
    polytensor_1_obj = polytensor.FirstDegreePolytensor(rank=1, tensor_data=scalar_data)
    polytensor_2_obj = polytensor.FirstDegreePolytensor(rank=1, tensor_data=matrices_data)
    assert_array_1 = [[10, 10], [10, 10]]
    for i in range(len(assert_array_1)):
        for j in range(len(assert_array_1)):
            assert assert_array_1[i][j] == polytensor_2_obj.multiply_matrix_scalar(polytensor_1_obj)[i][j]
    scalar_data = np.arange(56)
    matrices_data = np.zeros([56, 2, 2])
    matrices_data = np.full_like(matrices_data, 1)
    polytensor_1_obj = polytensor.FirstDegreePolytensor(rank=5, tensor_data=scalar_data)
    polytensor_2_obj = polytensor.FirstDegreePolytensor(rank=5, tensor_data=matrices_data)
    result = np.zeros([2, 2])
    for i in range(56):
        result += scalar_data[i] * matrices_data[i]
    assert np.allclose(result, polytensor_2_obj.multiply_matrix_scalar(polytensor_1_obj))


def test_truncate_tensor():
    tensor_data = np.arange(56)
    polytensor_1_obj = polytensor.FirstDegreePolytensor(rank=5, tensor_data=tensor_data)
    truncated_polytensor = polytensor_1_obj.truncate_tensor(4)
    assert truncated_polytensor.data.shape == (35,)
    truncated_polytensor = polytensor_1_obj.truncate_tensor(3)
    assert truncated_polytensor.data.shape == (20,)
    truncated_polytensor = polytensor_1_obj.truncate_tensor(2)
    assert truncated_polytensor.data.shape == (10,)
    truncated_polytensor = polytensor_1_obj.truncate_tensor(1)
    assert truncated_polytensor.data.shape == (4,)
    truncated_polytensor = polytensor_1_obj.truncate_tensor(0)
    assert truncated_polytensor.data.shape == (1,)


def test_multiply_first_degree_second_degree():
    polytensor_1_obj = polytensor.FirstDegreePolytensor(rank=1, tensor_data=np.array([1, 2, 3, 4]))
    polytensor_2_obj = polytensor.SecondDegreePolytensor(rank_1=1, rank_2=1, tensor_data=np.array([[1, 2, 3, 4],
                                                                                                   [5, 6, 7, 8],
                                                                                                   [9, 10, 11, 12],
                                                                                                   [13, 14, 15, 16]]))
    assert np.array_equal(polytensor_1_obj.multiply_first_degree_second_degree(polytensor_2_obj).data, polytensor_1_obj.
                          data.T @ polytensor_2_obj.data)
    polytensor_1_obj = polytensor.FirstDegreePolytensor(rank=1, tensor_data=np.array([1, 2, 3, 4]))
    polytensor_2_obj = polytensor.SecondDegreePolytensor(rank_1=1, rank_2=1, tensor_data=np.array([[1, 2, 3], [5, 6, 7],
                                                                                                   [9, 10, 11],
                                                                                                   [13, 14, 15]]))
    assert np.array_equal(polytensor_1_obj.multiply_first_degree_second_degree(polytensor_2_obj).data, polytensor_1_obj.
                          data.T @ polytensor_2_obj.data)
    tensor_1_data = np.arange(120)
    tensor_2_data = np.zeros([120, 120])
    tensor_2_data = np.full_like(tensor_2_data, 4)
    polytensor_1_obj = polytensor.FirstDegreePolytensor(rank=7, tensor_data=tensor_1_data)
    polytensor_2_obj = polytensor.SecondDegreePolytensor(rank_1=7, rank_2=7, tensor_data=tensor_2_data)
    assert np.array_equal(polytensor_1_obj.multiply_first_degree_second_degree(polytensor_2_obj).data, polytensor_1_obj.
                          data.T @ polytensor_2_obj.data)


def test_dot_first_degree():
    # Test case 1: Two 1D vectors with the same values
    a = polytensor.FirstDegreePolytensor(rank=1, tensor_data=np.array([1, 2, 3, 4]))
    b = polytensor.FirstDegreePolytensor(rank=1, tensor_data=np.array([1, 2, 3, 4]))
    expected_result = np.dot(a.data, b.data)
    assert np.array_equal(a.dot_first_degree(b), expected_result)
    # Test case 2: Two 1D vectors with different values
    a = polytensor.FirstDegreePolytensor(rank=1, tensor_data=np.array([1, 2, 3, 4]))
    b = polytensor.FirstDegreePolytensor(rank=1, tensor_data=np.array([5, 6, 7, 8]))
    expected_result = np.dot(a.data, b.data)
    assert np.array_equal(a.dot_first_degree(b), expected_result)
    # Test case 3: Two 1D vectors with different sizes
    a = polytensor.FirstDegreePolytensor(rank=1, tensor_data=np.array([1, 2, 3]))
    b = polytensor.FirstDegreePolytensor(rank=1, tensor_data=np.array([1, 2]))
    with pytest.raises(ValueError):
        a.dot_first_degree(b)


# Test SecondDegreePolytensor
def test_init_second_degree():
    polytensor_1_obj = polytensor.SecondDegreePolytensor(rank_1=5, rank_2=5)
    assert polytensor_1_obj.length_1 == 56
    assert np.shape(polytensor_1_obj.data) == (56, 56)
    polytensor_2_obj = polytensor.SecondDegreePolytensor(rank_1=3, rank_2=3)
    assert polytensor_2_obj.length_1 == 20
    assert np.shape(polytensor_2_obj.data) == (20, 20)
    polytensor_3_obj = polytensor.SecondDegreePolytensor(rank_1=0, rank_2=0)
    assert polytensor_3_obj.length_1 == 1
    assert np.shape(polytensor_3_obj.data) == (1, 1)
    polytensor_4_obj = polytensor.SecondDegreePolytensor(rank_1=5, rank_2=5)
    assert polytensor_4_obj._rank_2 == polytensor_4_obj._rank_1
    polytensor_5_obj = polytensor.SecondDegreePolytensor(rank_1=5, rank_2=4)
    assert polytensor_5_obj._rank_2 == 4 and polytensor_5_obj._rank_1 == 5
    assert polytensor_5_obj.length_2 == 35 and polytensor_5_obj.length_1 == 56
    polytensor_5_obj = polytensor.SecondDegreePolytensor(rank_1=5, rank_2=[4, 4])
    assert polytensor_5_obj.data.shape == (56, 15)
    tensor_1_data = np.zeros([120, 120])
    tensor_1_data = np.full_like(tensor_1_data, 4)
    polytensor_6_obj = polytensor.SecondDegreePolytensor(rank_1=7, rank_2=7, tensor_data=tensor_1_data)
    assert np.allclose(tensor_1_data, polytensor_6_obj.data)


def test_write_to_data_second_degree():
    tensor_1_data = np.zeros([120, 120])
    tensor_1_data = np.full_like(tensor_1_data, 4)
    polytensor_1_obj = polytensor.SecondDegreePolytensor(rank_1=7, rank_2=7)
    for i in range(tensor_1_data.shape[0]):
        for j in range(tensor_1_data.shape[1]):
            polytensor_1_obj.write_to_data(i, j, new_data=tensor_1_data[i, j])
    assert np.allclose(tensor_1_data, polytensor_1_obj.data)
    tensor_1_data = np.zeros([1, 1])
    tensor_1_data = np.full_like(tensor_1_data, 4)
    polytensor_1_obj = polytensor.SecondDegreePolytensor(rank_1=0, rank_2=0)
    for i in range(tensor_1_data.shape[0]):
        for j in range(tensor_1_data.shape[1]):
            polytensor_1_obj.write_to_data(i, j, new_data=tensor_1_data[i, j])
    assert np.allclose(tensor_1_data, polytensor_1_obj.data)


def test_write_interaction_tensor_multi_indices():
    polytensor_1_obj = polytensor.SecondDegreePolytensor(rank_1=0, rank_2=0, data_type=object)
    polytensor_1_obj.write_interaction_tensor_multi_indices()
    test_int_tensor = np.array([[np.array([0, 0, 0]), np.array([0, 0, 0])]], dtype=object)
    for i in range(len(test_int_tensor)):
        for j in range(len(test_int_tensor)):
            for k in range(2):
                for h in range(3):
                    assert test_int_tensor[i][j][k] == polytensor_1_obj.data[i][j][k][h]
    polytensor_2_obj = polytensor.SecondDegreePolytensor(rank_1=1, rank_2=1, data_type=object)
    polytensor_2_obj.write_interaction_tensor_multi_indices()
    test_int_tensor = np.array([[[np.array([0, 0, 0]), np.array([0, 0, 0])],
                                 [np.array([0, 0, 0]), np.array([1, 0, 0])],
                                 [np.array([0, 0, 0]), np.array([0, 1, 0])],
                                 [np.array([0, 0, 0]), np.array([0, 0, 1])]],
                                [[np.array([1, 0, 0]), np.array([0, 0, 0])],
                                 [np.array([1, 0, 0]), np.array([1, 0, 0])],
                                 [np.array([1, 0, 0]), np.array([0, 1, 0])],
                                 [np.array([1, 0, 0]), np.array([0, 0, 1])]],
                                [[np.array([0, 1, 0]), np.array([0, 0, 0])],
                                 [np.array([0, 1, 0]), np.array([1, 0, 0])],
                                 [np.array([0, 1, 0]), np.array([0, 1, 0])],
                                 [np.array([0, 1, 0]), np.array([0, 0, 1])]],
                                [[np.array([0, 0, 1]), np.array([0, 0, 0])],
                                 [np.array([0, 0, 1]), np.array([1, 0, 0])],
                                 [np.array([0, 0, 1]), np.array([0, 1, 0])],
                                 [np.array([0, 0, 1]), np.array([0, 0, 1])]]],
                               dtype=object)
    for i in range(len(test_int_tensor)):
        for j in range(len(test_int_tensor)):
            for k in range(2):
                for h in range(3):
                    assert test_int_tensor[i][j][k][h] == polytensor_2_obj.data[i][j][k][h]


def test_write_potential_tensor_multi_indices():
    polytensor_1_obj = polytensor.SecondDegreePolytensor(rank_1=0, rank_2=0, data_type=object)
    polytensor_1_obj.write_potential_tensor_multi_indices()
    test_int_tensor = np.array([[np.array([0, 0, 0]), np.array([0, 0, 0])]], dtype=object)
    for i in range(len(test_int_tensor)):
        for j in range(len(test_int_tensor)):
            for k in range(2):
                for h in range(3):
                    assert test_int_tensor[i][j][k] == polytensor_1_obj.data[i][j][k][h]
    polytensor_2_obj = polytensor.SecondDegreePolytensor(rank_1=1, rank_2=1, data_type=object)
    polytensor_2_obj.write_potential_tensor_multi_indices()
    test_int_tensor = np.array([[[np.array([0, 0, 0]), np.array([0, 0, 0])],
                                 [np.array([1, 0, 0]), np.array([0, 0, 0])],
                                 [np.array([0, 1, 0]), np.array([0, 0, 0])],
                                 [np.array([0, 0, 1]), np.array([0, 0, 0])]],
                                [[np.array([1, 0, 0]), np.array([0, 0, 0])],
                                 [np.array([2, 0, 0]), np.array([0, 0, 0])],
                                 [np.array([1, 1, 0]), np.array([0, 0, 0])],
                                 [np.array([1, 0, 1]), np.array([0, 0, 0])]],
                                [[np.array([0, 1, 0]), np.array([0, 0, 0])],
                                 [np.array([1, 1, 0]), np.array([0, 0, 0])],
                                 [np.array([0, 2, 0]), np.array([0, 0, 0])],
                                 [np.array([0, 1, 1]), np.array([0, 0, 0])]],
                                [[np.array([0, 0, 1]), np.array([0, 0, 0])],
                                 [np.array([1, 0, 1]), np.array([0, 0, 0])],
                                 [np.array([0, 1, 1]), np.array([0, 0, 0])],
                                 [np.array([0, 0, 2]), np.array([0, 0, 0])]]],
                               dtype=object)
    for i in range(len(test_int_tensor)):
        for j in range(len(test_int_tensor)):
            for k in range(2):
                for h in range(3):
                    assert test_int_tensor[i][j][k][h] == polytensor_2_obj.data[i][j][k][h]


def test_multiply_second_degree_first_degree():
    polytensor_1_obj = polytensor.FirstDegreePolytensor(rank=1, tensor_data=np.array([1, 2, 3, 4]))
    polytensor_2_obj = polytensor.SecondDegreePolytensor(rank_1=1, rank_2=1, tensor_data=np.array([[1, 2, 3, 4],
                                                                                                   [5, 6, 7, 8],
                                                                                                   [9, 10, 11, 12],
                                                                                                   [13, 14, 15, 16]]))
    assert np.allclose(polytensor_2_obj.multiply_second_degree_first_degree(polytensor_1_obj).data,
                       polytensor_2_obj.data @ polytensor_1_obj.data)
    polytensor_1_obj = polytensor.FirstDegreePolytensor(rank=1, tensor_data=np.array([1, 2, 3, 4]))
    polytensor_2_obj = polytensor.SecondDegreePolytensor(rank_1=1, rank_2=1, tensor_data=np.array([[1, 2, 3, 4],
                                                                                                   [5, 6, 7, 8],
                                                                                                   [9, 10, 11, 12]]))
    assert np.allclose(polytensor_2_obj.multiply_second_degree_first_degree(polytensor_1_obj).data,
                       polytensor_2_obj.data @ polytensor_1_obj.data)
    tensor_1_data = np.arange(120)
    tensor_2_data = np.zeros([120, 120])
    tensor_2_data = np.full_like(tensor_2_data, 4)
    polytensor_1_obj = polytensor.FirstDegreePolytensor(rank=7, tensor_data=tensor_1_data)
    polytensor_2_obj = polytensor.SecondDegreePolytensor(rank_1=7, rank_2=7, tensor_data=tensor_2_data)
    assert np.allclose(polytensor_2_obj.multiply_second_degree_first_degree(polytensor_1_obj).data,
                       polytensor_2_obj.data @ polytensor_1_obj.data)
    tensor_1_data = np.arange(35)
    tensor_2_data = np.zeros([56, 35])
    tensor_2_data = np.full_like(tensor_2_data, 4)
    polytensor_1_obj = polytensor.FirstDegreePolytensor(rank=4, tensor_data=tensor_1_data)
    polytensor_2_obj = polytensor.SecondDegreePolytensor(rank_1=5, rank_2=4, tensor_data=tensor_2_data)
    assert np.allclose(polytensor_2_obj.multiply_second_degree_first_degree(polytensor_1_obj).data,
                       polytensor_2_obj.data @ polytensor_1_obj.data)

@pytest.fixture
def initialized_statics():
    # Ensure constants are initialized before each test
    polytensor.statics = polytensor.PolytensorStatics()
    return polytensor.statics


def test_initialized_constants(initialized_statics):
    # Test that constants are initialized
    assert initialized_statics.initialized is True


def test_max_order_default_value():
    # Test default value of max_order
    values = polytensor.PolytensorStatics()
    assert values.max_order == 42


def test_max_order_custom_value():
    # Test custom value of max_order
    custom_max_order = 30
    values = polytensor.PolytensorStatics(max_order=custom_max_order)
    assert values.max_order == custom_max_order


def test_invalid_max_order():
    # Test when an invalid value is passed for max_order
    with pytest.raises(ValueError):
        polytensor.PolytensorStatics(max_order=-1)


def test_invalid_t_rank():
    # Test when an invalid value is passed for t_rank
    with pytest.raises(ValueError):
        polytensor.PolytensorStatics(t_rank=-1)


def test_interaction_tensor_non_empty(initialized_statics):
    # Test that interaction tensor is not empty
    assert initialized_statics.interaction_tensor_template.size > 0


def test_potential_tensor_non_empty(initialized_statics):
    # Test that potential tensor is not empty
    assert initialized_statics.potential_tensor_template.size > 0


def test_degeneracies_non_empty(initialized_statics):
    # Test that degeneracies data is not empty
    assert len(initialized_statics.degeneracies.data) > 0


def test_tensor_coefficients_non_empty(initialized_statics):
    # Test that tensor coefficients are not empty
    assert len(initialized_statics.tensor_coefficients) > 0


def test_factorials_non_empty(initialized_statics):
    # Test that factorials are not empty
    assert len(initialized_statics.factorials) > 0


def test_double_factorials_non_empty(initialized_statics):
    # Test that double factorials are not empty
    assert len(initialized_statics.double_factorials) > 0


def test_binomials_non_empty(initialized_statics):
    # Test that binomials are not empty
    assert initialized_statics.binomials.size > 0


def test_trinomials_non_empty(initialized_statics):
    # Test that trinomials are not empty
    assert initialized_statics.trinomials.size > 0


def test_trinomials_symmetry(initialized_statics):
    # Test symmetry of trinomial coefficients
    assert np.allclose(initialized_statics.trinomials, initialized_statics.trinomials.swapaxes(1, 2))


def test_statics_initialization():
    assert polytensor.statics.initialized is True


def test_interaction_tensor_shape():
    assert polytensor.statics.interaction_tensor_template.shape == (816, 816)


def test_potential_tensor_shape():
    assert polytensor.statics.potential_tensor_template.shape == (816, 816)


def test_interaction_tensor_values():
    assert np.allclose(polytensor.statics.interaction_tensor_template[0, 0],
                       [np.array([0, 0, 0]), np.array([0, 0, 0])])
    assert np.allclose(polytensor.statics.interaction_tensor_template[815, 815],
                       [np.array([0, 0, 15]), np.array([0, 0, 15])])


def test_potential_tensor_values():
    assert np.allclose(polytensor.statics.potential_tensor_template[0, 0],
                       [np.array([0, 0, 0]), np.array([0, 0, 0])])
    assert np.allclose(polytensor.statics.potential_tensor_template[815, 815],
                       [np.array([0, 0, 30]), np.array([0, 0, 0])])

def test_max_order():
    assert polytensor.statics.max_order == 42


def test_degeneracies_shape():
    assert len(polytensor.statics.degeneracies.data) == 14190
    assert polytensor.statics.degeneracies.data.shape == (14190,)


def test_tensor_coefficients_length():
    assert len(polytensor.statics.tensor_coefficients) == 43


def test_factorials_shape():
    assert polytensor.statics.factorials.shape == (43,)
    assert len(polytensor.statics.factorials) == 43


def test_double_factorials_shape():
    assert polytensor.statics.double_factorials.shape == (44,)
    assert len(polytensor.statics.double_factorials) == 44


def test_binomials_shape():
    assert polytensor.statics.binomials.shape == (127, 127)
    assert len(polytensor.statics.binomials) == 127


def test_trinomials_shape():
    assert polytensor.statics.trinomials.shape == (43, 43, 43)
    assert len(polytensor.statics.trinomials) == 43