import numpy as np
from pyframe.embedding import constants


def test_globals_init():
    # Initialize
    assert constants.Constants.initialized is False
    particle_globals = constants.values
    assert constants.values.initialized is True
    # Test values
    assert particle_globals.interaction_tensor_template.shape == (816, 816)
    assert np.allclose(particle_globals.interaction_tensor_template[0, 0], [np.array([0, 0, 0]), np.array([0, 0, 0])])
    assert np.allclose(particle_globals.potential_tensor_template[0, 0], [np.array([0, 0, 0]), np.array([0, 0, 0])])
    assert np.allclose(particle_globals.interaction_tensor_template[815, 815],
                       [np.array([0, 0, 15]), np.array([0, 0, 15])])
    assert np.allclose(particle_globals.potential_tensor_template[815, 815],
                       [np.array([0, 0, 30]), np.array([0, 0, 0])])
    assert particle_globals.max_order == 42
    assert len(particle_globals.degeneracies.data) == 14190
    assert particle_globals.degeneracies.data.shape == (14190,)
    assert len(particle_globals.tensor_coefficients) == 43
    assert particle_globals.factorials.shape == (43,)
    assert len(particle_globals.factorials) == 43
    assert particle_globals.double_factorials.shape == (44,)
    assert len(particle_globals.double_factorials) == 44
    assert particle_globals.binomials.shape == (127, 127)
    assert len(particle_globals.binomials) == 127
    assert particle_globals.trinomials.shape == (43, 43, 43)
    assert len(particle_globals.trinomials) == 43
