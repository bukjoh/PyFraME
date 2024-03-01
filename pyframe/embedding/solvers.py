import copy
import sys
import numpy as np

from mpi4py import MPI
from pyframe.embedding import constants, interaction_tensor
from typing import Tuple, Optional


def induced_dipoles_jacobi(coordinates: np.ndarray,
                           polarizabilities: np.ndarray,
                           exclusions: list,
                           indices: np.ndarray,
                           fields: np.ndarray,
                           starting_guess: np.ndarray,
                           threshold: float,
                           max_iterations: float,
                           comm: Optional[MPI.Comm] = None
                           ) -> Tuple[np.ndarray, int]:
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
                                             threshold=threshold,
                                             max_iterations=max_iterations)
    else:
        return induced_dipoles_jacobi_parallel(coordinates=coordinates,
                                               polarizabilities=polarizabilities,
                                               exclusions=exclusions,
                                               indices=indices,
                                               fields=fields,
                                               starting_guess=starting_guess,
                                               threshold=threshold,
                                               max_iterations=max_iterations,
                                               comm=comm)


def induced_dipoles_jacobi_serial(coordinates: np.ndarray,
                                  polarizabilities: np.ndarray,
                                  exclusions: list,
                                  indices: np.ndarray,
                                  fields: np.ndarray,
                                  starting_guess: np.ndarray,
                                  threshold: float,
                                  max_iterations: float
                                  ) -> Tuple[np.ndarray, int]:
    """Solves for dipoles that are induced in particle.Atoms with the element-based formula of the Jacobi method.

    Args:
        coordinates: Array of coordinates for all Atoms.
        polarizabilities: Array of polarizabilities for all Atoms.
        exclusions: List of exclusions for all Atoms.
        indices: Array of indices of the Atoms.
        fields: Array of static fields on all Atoms.
        starting_guess: Array of induced dipoles used as the starting guess.
        threshold: Convergence threshold for the residue norm between the (k+1)th and (k)th set of induced dipoles.
        max_iterations: Maximum number of iterations.

    Returns:
        ind_dipoles: Array of induced dipoles.
        new_fields: Array of fields originating from the induced dipoles.
        iteration: Number of iterations it took to converge the induced dipoles to the threshold.
    """
    # Calculate induced dipoles from other induced dipoles
    old_ind_dipoles = starting_guess
    residue_norm = sys.float_info.max
    iteration = 0
    new_fields = np.zeros([len(old_ind_dipoles), 3])
    ind_dipoles = np.zeros([len(fields), 3])
    while residue_norm > threshold:
        iteration += 1
        if iteration > max_iterations:
            raise RuntimeError("Did not converge after the maximum number of iterations.")
        for i, coordinate_i in enumerate(coordinates):
            ind_dipoles_fields = np.zeros(3)
            for j, coordinate_j in enumerate(coordinates):
                if indices[j] in exclusions[i]:
                    continue
                # Potential tensor template used
                ind_dipoles_fields += np.einsum('ij, j', interaction_tensor.
                                                compute_t_tensor(r_a=coordinate_j,
                                                                 r_b=coordinate_i,
                                                                 rank_a=1,
                                                                 rank_b=1,
                                                                 start_rank_a=1,
                                                                 start_rank_b=1,
                                                                 tensor_template=constants.values.
                                                                 potential_tensor_template).data,
                                                old_ind_dipoles[j])
            new_fields[i, :] = ind_dipoles_fields
        # Calculate total induced dipoles
        for i, new_field in enumerate(new_fields):
            ind_dipoles[i, :] = np.einsum('ij, j', polarizabilities[i], np.add(new_field, fields[i]))
        residue_norm = np.linalg.norm(ind_dipoles - old_ind_dipoles) / np.linalg.norm(old_ind_dipoles)
        old_ind_dipoles = copy.deepcopy(ind_dipoles)
    return ind_dipoles, iteration


def induced_dipoles_jacobi_parallel(coordinates: np.ndarray,
                                    polarizabilities: np.ndarray,
                                    exclusions: list,
                                    indices: np.ndarray,
                                    fields: np.ndarray,
                                    starting_guess: np.ndarray,
                                    threshold: float,
                                    max_iterations: float,
                                    comm: MPI.Comm = None
                                    ) -> Tuple[np.ndarray, int]:
    """Solves for dipoles that are induced in particle.Atoms with the element-based formula of the Jacobi method.

    Args:
        coordinates: Array of coordinates for all Atoms.
        polarizabilities: Array of polarizabilities for all Atoms.
        exclusions: List of exclusions for all Atoms.
        indices: Array of indices of the Atoms.
        fields: Array of static fields on all Atoms.
        starting_guess: Array of induced dipoles used as the starting guess.
        threshold: Convergence threshold for the residue norm between the (k+1)th and (k)th set of induced dipoles.
        max_iterations: Maximum number of iterations.
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
    residue_norm = sys.float_info.max
    iteration = 0
    new_fields_global = np.zeros([len(fields), 3])
    new_fields_local = np.zeros([len(fields), 3])
    ind_dipoles = np.zeros([len(fields), 3])
    while residue_norm > threshold:
        iteration += 1
        if iteration > max_iterations:
            raise RuntimeError("Did not converge after the maximum number of iterations.")
        for i in range(start, end):
            ind_dipoles_fields = np.zeros(3)
            for j, coordinate_j in enumerate(coordinates):
                if indices[j] in exclusions[i]:
                    continue
                ind_dipoles_fields += np.einsum('ij, j', interaction_tensor.
                                                compute_t_tensor(r_a=coordinate_j,
                                                                 r_b=coordinates[i],
                                                                 rank_a=1,
                                                                 rank_b=1,
                                                                 start_rank_a=1,
                                                                 start_rank_b=1,
                                                                 tensor_template=constants.values.
                                                                 potential_tensor_template).data,
                                                old_ind_dipoles[j])
            new_fields_local[i, :] = ind_dipoles_fields
        comm.Allreduce(new_fields_local, new_fields_global, op=MPI.SUM)
        for i, new_field in enumerate(new_fields_global):
            ind_dipoles[i, :] = np.einsum('ij, j', polarizabilities[i], np.add(new_field, fields[i]))
        residue_norm = np.linalg.norm(ind_dipoles - old_ind_dipoles) / np.linalg.norm(old_ind_dipoles)
        old_ind_dipoles = copy.deepcopy(ind_dipoles)
    return ind_dipoles, iteration
