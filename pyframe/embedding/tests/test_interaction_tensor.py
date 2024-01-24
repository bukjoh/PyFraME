"""Tests PyFraME.embedding.electrostatic_interactions.py"""
import numpy as np
from pyframe.embedding import interaction_tensor, constants
from pyframe.embedding.tests import test_electrostatic_interactions

oxygen_nucleus = test_electrostatic_interactions.oxygen_nucleus
oxygen_atom = test_electrostatic_interactions.oxygen_atom


def test_compute_t_tensor():
    # Test with interaction_tensor_template
    ref_potential = np.array(
        [0.75202669489040475, 6.3679443173877762e-2, 3.0699045537920477e-2, -4.1315647410858666e-6,
         9.5311929756771267e-3, 7.7985188788112876e-3, -1.0495468203592808e-6,
         -2.8858133904975198e-3, -5.0597310570715463e-7, -6.6453795851796043e-3,
         1.7845230927165381e-3, 2.4879519222560046e-3, -3.3483563607254773e-7,
         -9.6388595705967816e-5, -2.1422168821470319e-7, -1.6881344970105696e-3,
         -1.6741238607204253e-3, 6.2537563711756675e-9, -8.1382806153557845e-4,
         3.2858187970137201e-7], dtype=np.float64) / 8.0
    t_tensor = interaction_tensor.compute_t_tensor(r_a=oxygen_nucleus.coordinate,
                                                   r_b=oxygen_atom.coordinate,
                                                   rank_a=0,
                                                   rank_b=oxygen_atom.multipole_order,
                                                   tensor_template=constants.values.
                                                   interaction_tensor_template).data

    assert np.allclose(t_tensor, ref_potential)
    # Test with potential_tensor_template
    ref_potential = np.array(
        [0.75202669489040475, -6.3679443173877762e-2, -3.0699045537920477e-2, 4.1315647410858666e-6,
         9.5311929756771267e-3, 7.7985188788112876e-3, -1.0495468203592808e-6,
         -2.8858133904975198e-3, -5.0597310570715463e-7, -6.6453795851796043e-3,
         -1.7845230927165381e-3, -2.4879519222560046e-3, 3.3483563607254773e-7,
         9.6388595705967816e-5, 2.1422168821470319e-7, 1.6881344970105696e-3,
         1.6741238607204253e-3, -6.2537563711756675e-9, 8.1382806153557845e-4,
         -3.2858187970137201e-7], dtype=np.float64) / 8.0
    t_tensor = interaction_tensor.compute_t_tensor(r_a=oxygen_nucleus.coordinate,
                                                   r_b=oxygen_atom.coordinate,
                                                   rank_a=0,
                                                   rank_b=oxygen_atom.multipole_order,
                                                   tensor_template=constants.values.
                                                   potential_tensor_template).data
    assert np.allclose(t_tensor, ref_potential)
