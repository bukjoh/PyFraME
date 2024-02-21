import numpy as np

from mpi4py import MPI
from pyframe.embedding import constants, interaction_tensor, tensor_tools
from typing import Tuple, Optional


def induced_dipoles_jacobi(coordinates: np.ndarray,
                           polarizabilities: np.ndarray,
                           exclusions: list,
                           indices: np.ndarray,
                           fields: np.ndarray,
                           starting_guess: np.ndarray,
                           threshold: float,
                           comm: Optional[MPI.Comm] = None
                           ) -> Tuple[np.ndarray, np.ndarray, int]:
    if not isinstance(coordinates, np.ndarray) or not isinstance(polarizabilities, np.ndarray) or \
            not isinstance(exclusions, list) or not isinstance(indices, np.ndarray) or \
            not isinstance(fields, np.ndarray) or not isinstance(starting_guess, np.ndarray):
        raise ValueError("Wrong input format.")
    if comm is None:
        return induced_dipoles_jacobi_serial(coordinates=coordinates,
                                             polarizabilities=polarizabilities,
                                             exclusions=exclusions,
                                             indices=indices,
                                             fields=fields,
                                             starting_guess=starting_guess,
                                             threshold=threshold)
    else:
        return induced_dipoles_jacobi_parallel(coordinates=coordinates,
                                               polarizabilities=polarizabilities,
                                               exclusions=exclusions,
                                               indices=indices,
                                               fields=fields,
                                               starting_guess=starting_guess,
                                               threshold=threshold,
                                               comm=comm)


def induced_dipoles_jacobi_serial(coordinates: np.ndarray,
                                  polarizabilities: np.ndarray,
                                  exclusions: list,
                                  indices: np.ndarray,
                                  fields: np.ndarray,
                                  starting_guess: np.ndarray,
                                  threshold: float
                                  ) -> Tuple[np.ndarray, np.ndarray, int]:
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
                # Potential tensor template used
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
        residue_norm = tensor_tools.vec_residue_norm(ind_dipoles, old_ind_dipoles)
        old_ind_dipoles = ind_dipoles
    return ind_dipoles, new_fields, iteration


def induced_dipoles_jacobi_parallel(coordinates: np.ndarray,
                                    polarizabilities: np.ndarray,
                                    exclusions: list,
                                    indices: np.ndarray,
                                    fields: np.ndarray,
                                    starting_guess: np.ndarray,
                                    threshold: float,
                                    comm: Optional[MPI.Comm] = None
                                    ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Solves for dipoles that are induced in particle.Atoms with the element-based formula of the Jacobi method.

    Args:
        coordinates: Array of coordinates for all Atoms.
        polarizabilities: Array of polarizabilities for all Atoms.
        exclusions: List of exclusions for all Atoms.
        indices: Array of indices of the Atoms.
        fields: Array of static fields on all Atoms.
        starting_guess: Array of induced dipoles used as the starting guess.
        threshold: Convergence threshold for the residue norm between the (k+1)th and (k)th set of induced dipoles.
        comm: The MPI communicator.

    Returns:
        ind_dipoles: Array of induced dipoles.
        new_fields: Array of fields originating from the induced dipoles.
        iteration: Number of iterations it took to converge the induced dipoles to the threshold.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    avg, res = divmod(len(coordinates), size)
    counts = [avg + 1 if p < res else avg for p in range(size)]
    start = sum(counts[:rank])
    end = sum(counts[:rank + 1])
    # Calculate induced dipoles from other induced dipoles
    old_ind_dipoles = starting_guess
    ind_dipoles = np.zeros([len(fields), 3])
    residue_norm = 1.
    iteration = 0
    new_fields_global = None
    while residue_norm > threshold:
        iteration += 1
        new_fields_global = np.zeros([len(old_ind_dipoles), 3])
        new_fields_local = np.zeros([len(old_ind_dipoles), 3])
        for i in range(start, end):
            field_component = np.zeros(3)
            for j, coordinate_j in enumerate(coordinates):
                if indices[j] in exclusions[i]:
                    continue
                field_component += np.einsum('ij, j', interaction_tensor.
                                             compute_t_tensor(r_a=coordinate_j,
                                                              r_b=coordinates[i],
                                                              rank_a=1,
                                                              rank_b=1,
                                                              start_rank_a=1,
                                                              start_rank_b=1,
                                                              tensor_template=constants.values.
                                                              potential_tensor_template).data,
                                             old_ind_dipoles[j])
            new_fields_local[i, :] = field_component
        comm.Allreduce(new_fields_local, new_fields_global, op=MPI.SUM)
        for i, new_field in enumerate(new_fields_global):
            ind_dipoles[i, :] = np.einsum('ij, j',
                                          polarizabilities[i], np.add(new_field,
                                                                      fields[i]))

        residue_norm = tensor_tools.vec_residue_norm(ind_dipoles, old_ind_dipoles)
        old_ind_dipoles = ind_dipoles

    return ind_dipoles, new_fields_global, iteration
