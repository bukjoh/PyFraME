import numpy as np
import copy
from pyframe.embedding import constants, interaction_tensor

def induced_dipoles_jacobi(coordinates: np.ndarray,
                           polarizabilities: np.ndarray,
                           exclusions: list,
                           indices: np.ndarray,
                           fields: np.ndarray,
                           starting_guess: np.ndarray,
                           threshold: float):
    """Solves for dipoles that are induced in particle.Atoms with the element-based formula of the Jacobi method.

    Args:
        coordinates: Array of coordinates for all Atoms.
        polarizabilities: Array of polarizabilities for all Atoms.
        exclusions: List of exclusions for all Atoms.
        indices: Array of indices of the Atoms.
        fields: Array of static fields on all Atoms.
        starting_guess: Array of induced dipoles used as the starting guess.
        threshold: Convergence threshold for the residue norm between the (k+1)th and (k)th set of induced dipoles.

    Returns:
        ind_dipoles: Array of induced dipoles.
        new_fields: Array of fields originating from the induced dipoles.
        iteration: Number of iterations it took to converge the induced dipoles to the threshold.
    """
    if not isinstance(coordinates, np.ndarray) or not isinstance(polarizabilities, np.ndarray) or \
            not isinstance(exclusions, list) or not isinstance(indices, np.ndarray) or \
            not isinstance(fields, np.ndarray) or not isinstance(starting_guess, np.ndarray):
        raise ValueError("Wrong input format.")

    # Calculate induced dipoles from other induced dipoles
    old_ind_dipoles = starting_guess
    ind_dipoles = np.zeros([len(fields), 3])
    residue_norm = 1.
    iteration = 0
    new_fields = None
    while residue_norm > threshold:
        iteration += 1
        new_fields = np.zeros([len(old_ind_dipoles), 3])
        for i, coordinate_i in enumerate(coordinates):
            field_component = np.zeros(3)
            for j, coordinate_j in enumerate(coordinates):
                if indices[j] in exclusions[i]:
                    continue
                # Changed template to potential rather than interaction! could be wrong though..
                field_component += np.einsum('ij, j', interaction_tensor.
                                             compute_t_tensor(r_a=coordinate_j,
                                                              r_b=coordinate_i,
                                                              rank_a=1,
                                                              rank_b=1,
                                                              start_rank_a=1,
                                                              start_rank_b=1,
                                                              tensor_template=constants.values.
                                                              potential_tensor_template).data,
                                             old_ind_dipoles[j])
            new_fields[i, :] = field_component
        # Calculate total induced dipoles
        for i, new_field in enumerate(new_fields):
            ind_dipoles[i, :] = np.einsum('ij, j',
                                          polarizabilities[i], np.add(new_field,
                                                                      fields[i]))
        residue_norm = np.abs(np.linalg.norm(ind_dipoles - old_ind_dipoles) / np.linalg.norm(old_ind_dipoles))
        old_ind_dipoles = copy.deepcopy(ind_dipoles)
    return ind_dipoles, new_fields, iteration