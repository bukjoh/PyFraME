from __future__ import annotations

import numpy as np
try:
    from pyframe.embedding import cpp_interaction_tensor_element
    print('Using cpp_interaction_tensor_element.')
    cpp_tool = True
except ImportError:
    print('Unable to import cpp_interaction_tensor_element. Please compile interaction_tensor_element.cpp. Using Python'
          ' version instead.')
    from pyframe.embedding import tensor_tools
    cpp_tool = False
from pyframe.embedding import polytensor, constants
from typing import Optional


def compute_t_tensor(r_a: np.ndarray,
                     r_b: np.ndarray,
                     rank_a: int,
                     rank_b: int,
                     tensor_template: np.ndarray,
                     start_rank_b: Optional[int] = 0,
                     start_rank_a: Optional[int] = 0
                     ) -> polytensor.SecondDegreePolytensor:
    """Builds and calculates the T tensor, which is a Matrix used to calculate the potential and interaction energy of
    Particles.

    Args:
        r_a: Cartesian coordinates of the first Particle.
        r_b: Cartesian coordinates of the second Particle.
        rank_a: Maximum column rank of the SecondDegreePolytensor.
        rank_b: Maximum row rank of the SecondDegreePolytensor.
        tensor_template: Template that contains the multi-indices to be calculated. Note that for calculating
        derivatives of the potential, and interaction energies the template should be different.
        start_rank_a: Minimum column rank of the SecondDegreePolytensor.
        start_rank_b: Minimum row rank of the SecondDegreePolytensor.

    Returns:
        T tensor as a SecondDegreePolytensor.
        (See Jon Applequist J. Math. Phys. 24, 736 (1983) for details on Polytensors.)
    """
    r_ab = r_b - r_a
    if r_a[0] == r_b[0] and r_a[1] == r_b[1] and r_a[2] == r_b[2]:
        raise ValueError("r_a and r_b cannot be equal.")
    interaction_tensor = polytensor.SecondDegreePolytensor(rank_2=[start_rank_b, rank_b],
                                                           rank_1=[start_rank_a, rank_a])
    start_b = (start_rank_b - 1 + 1) * (start_rank_b - 1 + 2) * (start_rank_b - 1 + 3) // 6
    end_b = (rank_b + 1) * (rank_b + 2) * (rank_b + 3) // 6
    start_a = (start_rank_a - 1 + 1) * (start_rank_a - 1 + 2) * (start_rank_a - 1 + 3) // 6
    end_a = (rank_a + 1) * (rank_a + 2) * (rank_a + 3) // 6
    if cpp_tool:
        for i in range(start_a, end_a):
            for j in range(start_b, end_b):
                interaction_element = cpp_interaction_tensor_element.compute_interaction_tensor_element(
                    tensor_template[i, j], r_ab, constants.values.tensor_coefficients)
                interaction_tensor.write_to_data(i=i - start_a, j=j - start_b, new_data=interaction_element)
    else:
        for i in range(start_a, end_a):
            for j in range(start_b, end_b):
                interaction_element = tensor_tools.compute_interaction_tensor_element(distance_vector=r_ab,
                                                                                      multi_index=tensor_template[i, j],
                                                                                      tensor_coefficients=constants.
                                                                                      values.tensor_coefficients)
                interaction_tensor.write_to_data(i=i - start_a, j=j - start_b, new_data=interaction_element)
    return interaction_tensor
