"""Tests PyFraME.embedding.constants.py"""
import pytest
import numpy as np

from pyframe.embedding import constants


@pytest.fixture
def initialized_constants():
    # Ensure constants are initialized before each test
    constants.values = constants.Constants()
    return constants.values

def test_initialized_constants(initialized_constants):
    # Test that constants are initialized
    assert initialized_constants.initialized is True

def test_max_order_default_value():
    # Test default value of max_order
    values = constants.Constants()
    assert values.max_order == 42

def test_max_order_custom_value():
    # Test custom value of max_order
    custom_max_order = 30
    values = constants.Constants(max_order=custom_max_order)
    assert values.max_order == custom_max_order

def test_invalid_max_order():
    # Test when an invalid value is passed for max_order
    with pytest.raises(ValueError):
        constants.Constants(max_order=-1)

def test_invalid_t_rank():
    # Test when an invalid value is passed for t_rank
    with pytest.raises(ValueError):
        constants.Constants(t_rank=-1)

def test_interaction_tensor_non_empty(initialized_constants):
    # Test that interaction tensor is not empty
    assert initialized_constants.interaction_tensor_template.size > 0

def test_potential_tensor_non_empty(initialized_constants):
    # Test that potential tensor is not empty
    assert initialized_constants.potential_tensor_template.size > 0

def test_degeneracies_non_empty(initialized_constants):
    # Test that degeneracies data is not empty
    assert len(initialized_constants.degeneracies.data) > 0

def test_tensor_coefficients_non_empty(initialized_constants):
    # Test that tensor coefficients are not empty
    assert len(initialized_constants.tensor_coefficients) > 0

def test_factorials_non_empty(initialized_constants):
    # Test that factorials are not empty
    assert len(initialized_constants.factorials) > 0

def test_double_factorials_non_empty(initialized_constants):
    # Test that double factorials are not empty
    assert len(initialized_constants.double_factorials) > 0

def test_binomials_non_empty(initialized_constants):
    # Test that binomials are not empty
    assert initialized_constants.binomials.size > 0

def test_trinomials_non_empty(initialized_constants):
    # Test that trinomials are not empty
    assert initialized_constants.trinomials.size > 0

def test_trinomials_symmetry(initialized_constants):
    # Test symmetry of trinomial coefficients
    assert np.allclose(initialized_constants.trinomials, initialized_constants.trinomials.swapaxes(1, 2))

def test_constants_initialization():
    assert constants.values.initialized is True

def test_interaction_tensor_shape():
    assert constants.values.interaction_tensor_template.shape == (816, 816)

def test_potential_tensor_shape():
    assert constants.values.potential_tensor_template.shape == (816, 816)

def test_interaction_tensor_values():
    assert np.allclose(constants.values.interaction_tensor_template[0, 0],
                       [np.array([0, 0, 0]), np.array([0, 0, 0])])
    assert np.allclose(constants.values.interaction_tensor_template[815, 815],
                       [np.array([0, 0, 15]), np.array([0, 0, 15])])

def test_potential_tensor_values():
    assert np.allclose(constants.values.potential_tensor_template[0, 0],
                       [np.array([0, 0, 0]), np.array([0, 0, 0])])
    assert np.allclose(constants.values.potential_tensor_template[815, 815],
                       [np.array([0, 0, 30]), np.array([0, 0, 0])])

def test_max_order():
    assert constants.values.max_order == 42

def test_degeneracies_shape():
    assert len(constants.values.degeneracies.data) == 14190
    assert constants.values.degeneracies.data.shape == (14190,)

def test_tensor_coefficients_length():
    assert len(constants.values.tensor_coefficients) == 43

def test_factorials_shape():
    assert constants.values.factorials.shape == (43,)
    assert len(constants.values.factorials) == 43

def test_double_factorials_shape():
    assert constants.values.double_factorials.shape == (44,)
    assert len(constants.values.double_factorials) == 44

def test_binomials_shape():
    assert constants.values.binomials.shape == (127, 127)
    assert len(constants.values.binomials) == 127

def test_trinomials_shape():
    assert constants.values.trinomials.shape == (43, 43, 43)
    assert len(constants.values.trinomials) == 43
