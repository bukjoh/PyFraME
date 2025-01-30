import copy
import sys
import numpy as np
import scipy
import sklearn.cluster

from mpi4py import MPI
from pyframe.embedding import engine
from typing import Tuple, Optional


def induced_dipoles_fmm(coordinates: np.ndarray,
                        polarizabilities: np.ndarray,
                        exclusions: list,
                        indices: np.ndarray,
                        fields: np.ndarray,
                        starting_guess: np.ndarray,
                        mic: bool,
                        box: np.ndarray,
                        threshold: float,
                        max_iterations: Optional[int] = 100,
                        tree_ncrit: int = 64,
                        tree_expansion_order: int = 5,
                        theta: float = 0.5,
                        comm: Optional[MPI.Comm] = None
                        ) -> Tuple[np.ndarray, int]:
    """Solves for dipoles that are induced in particle.Atoms with the element-based formula of the Jacobi method either
    in a serial or a parallel computation scheme.

    Args:
        coordinates: Array of coordinates for all Atoms.
        polarizabilities: Array of polarizabilities for all Atoms.
        exclusions: List of exclusions for all Atoms.
        indices: Array of atom indices.
        fields: Array of static fields on all Atoms.
        starting_guess: Array of induced dipoles used as the starting guess.
        mic: Boolean indicating whether induced dipoles are calculated using mic scaled coordinates.
        box: box dimensions as described by three vectors spanning the box.
        threshold: Convergence threshold for the residue norm between the (k+1)th and (k)th set of induced dipoles.
        max_iterations: Maximum number of iterations.
        tree_ncrit: FMM parameter: Maximum number of particles per tree node.
        tree_expansion_order: FMM parameter: Expansion order for tree-based summation schemes.
        theta: FMM parameter: Opening angle for tree-based summation schemes.
        comm: The MPI communicator.

    Returns:
        ind_dipoles: Array of induced dipoles.
        new_fields: Array of fields originating from the induced dipoles.
        iteration: Number of iterations it took to converge the induced dipoles to the threshold.
    """
    if not isinstance(coordinates, np.ndarray) or not isinstance(polarizabilities, np.ndarray) or \
            not isinstance(exclusions, list) or not isinstance(indices, np.ndarray) or \
            not isinstance(fields, np.ndarray) or not isinstance(starting_guess, np.ndarray):
        raise ValueError("Wrong input format.")
    shifted_exclusions = [
        tuple(value - 1 for value in exclusion) for exclusion in exclusions
    ]
    if mic:
        engine.set_mic_coords_idcs_exlcs(coordinates, indices, shifted_exclusions, box)
    else:
        engine.set_coords_idcs_exlcs(coordinates, indices, shifted_exclusions)

    # TODO enable MIC
    # Calculate induced dipoles from other induced dipoles
    old_ind_dipoles = starting_guess
    residue_norm = sys.float_info.max
    max_residue_norm = sys.float_info.max
    iteration = 0
    ind_dipoles = np.zeros([len(fields), 3], dtype=np.float64)

    while not (residue_norm < threshold and max_residue_norm < threshold):
        iteration += 1
        engine.set_old_ind_dipoles(old_ind_dipoles)
        if iteration > max_iterations:
            raise RuntimeError("Did not converge after the maximum number of iterations.")
        # TODO remove damping.
        damping = 0.0
        new_fields = -1 * engine.ind_dipoles_fields_fmm(tree_ncrit, tree_expansion_order, theta, damping)
        for i, new_field in enumerate(new_fields):
            ind_dipoles[i, :] = np.einsum('ij, j', polarizabilities[i], (new_field + fields[i]))
        residue_norm = np.linalg.norm(ind_dipoles - old_ind_dipoles)
        max_residue_norm = np.max(np.abs(ind_dipoles - old_ind_dipoles))
        old_ind_dipoles = copy.deepcopy(ind_dipoles)

    return ind_dipoles, iteration


def induced_dipoles_jacobi(coordinates: np.ndarray,
                           polarizabilities: np.ndarray,
                           exclusions: list,
                           indices: np.ndarray,
                           fields: np.ndarray,
                           starting_guess: np.ndarray,
                           mic: bool,
                           box: np.ndarray,
                           threshold: float,
                           max_iterations: Optional[int] = 100,
                           comm: Optional[MPI.Comm] = None
                           ) -> Tuple[np.ndarray, int]:
    """Solves for dipoles that are induced in particle.Atoms with the element-based formula of the Jacobi method either
    in a serial or a parallel computation scheme.

    Args:
        coordinates: Array of coordinates for all Atoms.
        polarizabilities: Array of polarizabilities for all Atoms.
        exclusions: List of exclusions for all Atoms.
        indices: Array of atom indices.
        fields: Array of static fields on all Atoms.
        starting_guess: Array of induced dipoles used as the starting guess.
        mic: Boolean indicating whether induced dipoles are calculated using mic scaled coordinates.
        box: box dimensions as described by three vectors spanning the box.
        threshold: Convergence threshold for the residue norm between the (k+1)th and (k)th set of induced dipoles.
        max_iterations: Maximum number of iterations.
        comm: The MPI communicator.

    Returns:
        ind_dipoles: Array of induced dipoles.
        new_fields: Array of fields originating from the induced dipoles.
        iteration: Number of iterations it took to converge the induced dipoles to the threshold.
    """
    if not isinstance(coordinates, np.ndarray) or not isinstance(polarizabilities, np.ndarray) or \
            not isinstance(exclusions, list) or not isinstance(indices, np.ndarray) or \
            not isinstance(fields, np.ndarray) or not isinstance(starting_guess, np.ndarray):
        raise ValueError("Wrong input format.")
    if mic:
        engine.set_mic_coords_idcs_exlcs(coordinates, indices, exclusions, box)
    else:
        engine.set_coords_idcs_exlcs(coordinates, indices, exclusions)
    if comm is None:
        return induced_dipoles_jacobi_serial(polarizabilities=polarizabilities,
                                             fields=fields,
                                             starting_guess=starting_guess,
                                             mic=mic,
                                             threshold=threshold,
                                             max_iterations=max_iterations)
    else:
        return induced_dipoles_jacobi_parallel(polarizabilities=polarizabilities,
                                               fields=fields,
                                               starting_guess=starting_guess,
                                               mic=mic,
                                               threshold=threshold,
                                               max_iterations=max_iterations,
                                               comm=comm)


def induced_dipoles_jacobi_serial(polarizabilities: np.ndarray,
                                  fields: np.ndarray,
                                  starting_guess: np.ndarray,
                                  mic: bool,
                                  threshold: float,
                                  max_iterations: int
                                  ) -> Tuple[np.ndarray, int]:
    """Solves for dipoles that are induced in particle.Atoms with the element-based formula of the Jacobi method.

    Args:
        polarizabilities: Array of polarizabilities for all Atoms.
        fields: Array of static fields on all Atoms.
        starting_guess: Array of induced dipoles used as the starting guess.
        mic: Boolean indicating whether induced dipoles are calculated using mic scaled coordinates.
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
    max_residue_norm = sys.float_info.max
    iteration = 0
    ind_dipoles = np.zeros([len(fields), 3], dtype=np.float64)
    while not (residue_norm < threshold and max_residue_norm < threshold):
        iteration += 1
        engine.set_old_ind_dipoles(old_ind_dipoles)
        if iteration > max_iterations:
            raise RuntimeError("Did not converge after the maximum number of iterations.")
        new_fields = engine.ind_dipoles_fields(np.array([0, len(fields), mic], dtype=np.int64))
        # Calculate total induced dipoles
        for i, new_field in enumerate(new_fields):
            ind_dipoles[i, :] = np.einsum('ij, j', polarizabilities[i], (new_field + fields[i]))
        residue_norm = np.linalg.norm(ind_dipoles - old_ind_dipoles)
        max_residue_norm = np.max(np.abs(ind_dipoles - old_ind_dipoles))
        old_ind_dipoles = copy.deepcopy(ind_dipoles)
    return ind_dipoles, iteration


def induced_dipoles_jacobi_parallel(polarizabilities: np.ndarray,
                                    fields: np.ndarray,
                                    starting_guess: np.ndarray,
                                    mic: bool,
                                    threshold: float,
                                    max_iterations: int,
                                    comm: MPI.Comm = None
                                    ) -> Tuple[np.ndarray, int]:
    """Solves for dipoles that are induced in particle.Atoms with the element-based formula of the Jacobi method.

    Args:
        polarizabilities: Array of polarizabilities for all Atoms.
        fields: Array of static fields on all Atoms.
        starting_guess: Array of induced dipoles used as the starting guess.
        mic: Boolean indicating whether induced dipoles are calculated using mic scaled coordinates.
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
    avg, res = divmod(len(fields), size)
    counts = [avg + 1 if p < res else avg for p in range(size)]
    start = sum(counts[:rank])
    end = sum(counts[:rank + 1])
    # Calculate induced dipoles from other induced dipoles
    old_ind_dipoles = starting_guess
    residue_norm = sys.float_info.max
    max_residue_norm = sys.float_info.max
    iteration = 0
    ind_dipoles = np.zeros([len(fields), 3], dtype=np.float64)
    new_fields_global = np.zeros([len(fields), 3], dtype=np.float64)
    while not (residue_norm < threshold and max_residue_norm < threshold):
        iteration += 1
        engine.set_old_ind_dipoles(old_ind_dipoles)
        if iteration > max_iterations:
            raise RuntimeError("Did not converge after the maximum number of iterations.")
        # TODO optimize this parts parallelization + maybe general consideration to move this into the c++ layer?
        new_fields_local = engine.ind_dipoles_fields(np.array([start, end, mic], dtype=np.int64))
        comm.Allreduce(new_fields_local, new_fields_global, op=MPI.SUM)
        # Calculate total induced dipoles
        for i, new_field in enumerate(new_fields_global):
            ind_dipoles[i, :] = np.einsum('ij, j', polarizabilities[i], np.add(new_field, fields[i]))
        residue_norm = np.linalg.norm(ind_dipoles - old_ind_dipoles)
        max_residue_norm = np.max(np.abs(ind_dipoles - old_ind_dipoles))
        old_ind_dipoles = copy.deepcopy(ind_dipoles)
    return ind_dipoles, iteration


def induced_dipoles_jidiis(coordinates: np.ndarray,
                           polarizabilities: np.ndarray,
                           exclusions: list,
                           indices: np.ndarray,
                           fields: np.ndarray,
                           starting_guess: np.ndarray,
                           mic: bool,
                           box: np.ndarray,
                           threshold: float,
                           max_iterations: Optional[int] = 100,
                           init_diis: Optional[int] = 3,
                           max_diis: Optional[int] = 10,
                           comm: Optional[MPI.Comm] = None
                           ) -> Tuple[np.ndarray, int]:
    """Solves for dipoles that are induced in particle.Atoms with the element-based formula of the JIDIIS method.
    JIDIIS is based on the Jacobi method, with direct inversion in the iterative subspace to improve convergence.

    Args:
        coordinates: Array of coordinates for all Atoms.
        polarizabilities: Array of polarizabilities for all Atoms.
        exclusions: List of exclusions for all Atoms.
        indices: Array of atom indices.
        fields: Array of static fields on all Atoms.
        starting_guess: Array of induced dipoles used as the starting guess.
        mic: Boolean indicating whether induced dipoles are calculated using mic scaled coordinates.
        box: box dimensions as described by three vectors spanning the box.
        threshold: Convergence threshold for the residue norm between the (k+1)th and (k)th set of induced dipoles.
        max_iterations: Maximum number of iterations.
        max_diis: Maximum number of previous iterations to consider in the DIIS method.
        init_diis: Iteration number at which DIIS is initiated.
        comm: The MPI communicator.

    Returns:
        ind_dipoles: Array of induced dipoles.
        new_fields: Array of fields originating from the induced dipoles.
        iteration: Number of iterations it took to converge the induced dipoles to the threshold.
    """
    if not isinstance(coordinates, np.ndarray) or not isinstance(polarizabilities, np.ndarray) or \
            not isinstance(exclusions, list) or not isinstance(indices, np.ndarray) or \
            not isinstance(fields, np.ndarray) or not isinstance(starting_guess, np.ndarray):
        raise ValueError("Wrong input format.")
    if mic:
        engine.set_mic_coords_idcs_exlcs(coordinates, indices, exclusions, box)
    else:
        engine.set_coords_idcs_exlcs(coordinates, indices, exclusions)
    if comm is None:
        return induced_dipoles_jidiis_serial(polarizabilities=polarizabilities,
                                             fields=fields,
                                             starting_guess=starting_guess,
                                             mic=mic,
                                             threshold=threshold,
                                             max_iterations=max_iterations,
                                             init_diis=init_diis,
                                             max_diis=max_diis)
    else:
        return induced_dipoles_jidiis_parallel(polarizabilities=polarizabilities,
                                               fields=fields,
                                               starting_guess=starting_guess,
                                               mic=mic,
                                               threshold=threshold,
                                               max_iterations=max_iterations,
                                               init_diis=init_diis,
                                               max_diis=max_diis,
                                               comm=comm)


def induced_dipoles_jidiis_serial(polarizabilities: np.ndarray,
                                  fields: np.ndarray,
                                  starting_guess: np.ndarray,
                                  mic: bool,
                                  threshold: float,
                                  max_iterations: int,
                                  init_diis: int,
                                  max_diis: int
                                  ) -> Tuple[np.ndarray, int]:
    """Solves for dipoles that are induced in particle.Atoms with the element-based formula of the JIDIIS method.
    JIDIIS is based on the Jacobi method, with direct inversion in the iterative subspace to improve convergence.

    Args:
        polarizabilities: Array of polarizabilities for all Atoms.
        fields: Array of static fields on all Atoms.
        starting_guess: Array of induced dipoles used as the starting guess.
        mic: Boolean indicating whether induced dipoles are calculated using mic scaled coordinates.
        threshold: Convergence threshold for the residue norm between the (k+1)th and (k)th set of induced dipoles.
        max_iterations: Maximum number of iterations.
        max_diis: Maximum number of previous iterations to consider in the DIIS method.
        init_diis: Iteration number at which DIIS is initiated.

    Returns:
        ind_dipoles: Array of induced dipoles.
        new_fields: Array of fields originating from the induced dipoles.
        iteration: Number of iterations it took to converge the induced dipoles to the threshold.
    """
    # Calculate induced dipoles from other induced dipoles
    old_ind_dipoles = starting_guess
    ind_dipoles = starting_guess
    residue = np.array([0], dtype=np.float64)
    iteration = 0
    new_ind_dipoles = np.zeros([len(fields), 3], dtype=np.float64)
    error_matrix_full = np.ones((max_iterations + 1, max_iterations + 1), dtype=np.float64)
    error_matrix_full = -error_matrix_full
    error_matrix_full = error_matrix_full + np.diag(np.ones(max_iterations + 1, dtype=np.float64))
    while iteration <= max_iterations:
        iteration += 1
        engine.set_old_ind_dipoles(old_ind_dipoles)
        new_fields = engine.ind_dipoles_fields(np.array([0, len(fields), mic], dtype=np.int64))
        # Calculate total induced dipoles
        for i, new_field in enumerate(new_fields):
            new_ind_dipoles[i, :] = np.einsum('ij, j', polarizabilities[i], np.add(new_field, fields[i]))
        new_residue = new_ind_dipoles - old_ind_dipoles
        residue_norm = np.linalg.norm(new_residue)
        max_residue_norm = np.max(np.abs(new_residue))
        # Check convergence
        if residue_norm < threshold and max_residue_norm < threshold:
            break
        # Save residue and dipoles
        if iteration == 1:
            residue = np.expand_dims(new_residue, axis=2)
            ind_dipoles = np.expand_dims(ind_dipoles, axis=2)
        else:
            residue = np.concatenate((residue, np.expand_dims(new_residue, axis=2)), axis=2)
        ind_dipoles = np.concatenate((ind_dipoles, np.expand_dims(new_ind_dipoles, axis=2)), axis=2)
        # Add to error matrix
        for i in range(0, iteration):
            error = np.sum(residue[:, :, i] * residue[:, :, iteration - 1])
            error_matrix_full[i, iteration - 1] = error
            error_matrix_full[iteration - 1, i] = error
        # Start diis at the assigned iteration
        if iteration >= init_diis:
            new_ind_dipoles = direct_inversion_iterative_subspace(ind_dipoles=ind_dipoles,
                                                                  error_matrix_full=error_matrix_full,
                                                                  max_diis=max_diis,
                                                                  iteration=iteration)
        old_ind_dipoles = copy.deepcopy(new_ind_dipoles)
    # Check maximum convergence
    if iteration > max_iterations:
        raise RuntimeError("Did not converge after the maximum number of iterations.")

    return old_ind_dipoles, iteration


def induced_dipoles_jidiis_parallel(polarizabilities: np.ndarray,
                                    fields: np.ndarray,
                                    starting_guess: np.ndarray,
                                    mic: bool,
                                    threshold: float,
                                    max_iterations: int,
                                    init_diis: int,
                                    max_diis: int,
                                    comm: MPI.Comm = None
                                    ) -> Tuple[np.ndarray, int]:
    """Solves for dipoles that are induced in particle.Atoms with the element-based formula of the JIDIIS method.
    JIDIIS is based on the Jacobi method, with direct inversion in the iterative subspace to improve convergence.

    Args:
        polarizabilities: Array of polarizabilities for all Atoms.
        fields: Array of static fields on all Atoms.
        starting_guess: Array of induced dipoles used as the starting guess.
        mic: Boolean indicating whether induced dipoles are calculated using mic scaled coordinates.
        threshold: Convergence threshold for the residue norm between the (k+1)th and (k)th set of induced dipoles.
        max_iterations: Maximum number of iterations.
        max_diis: Maximum number of previous iterations to consider in the DIIS method.
        init_diis: Iteration number at which DIIS is initiated.
        comm: The MPI communicator.

    Returns:
        ind_dipoles: Array of induced dipoles.
        new_fields: Array of fields originating from the induced dipoles.
        iteration: Number of iterations it took to converge the induced dipoles to the threshold.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    avg, res = divmod(len(fields), size)
    counts = [avg + 1 if p < res else avg for p in range(size)]
    start = sum(counts[:rank])
    end = sum(counts[:rank + 1])
    # Calculate induced dipoles from other induced dipoles
    old_ind_dipoles = starting_guess
    ind_dipoles = starting_guess
    residue = np.array([0], dtype=np.float64)
    iteration = 0
    new_ind_dipoles = np.zeros([len(fields), 3], dtype=np.float64)
    new_fields_global = np.zeros([len(fields), 3], dtype=np.float64)
    error_matrix_full = np.ones((max_iterations + 1, max_iterations + 1), dtype=np.float64)
    error_matrix_full = -error_matrix_full
    error_matrix_full = error_matrix_full + np.diag(np.ones(max_iterations + 1, dtype=np.float64))
    while iteration <= max_iterations:
        iteration += 1
        engine.set_old_ind_dipoles(old_ind_dipoles)
        new_fields_local = engine.ind_dipoles_fields(np.array([start, end, mic], dtype=np.int64))
        comm.Allreduce(new_fields_local, new_fields_global, op=MPI.SUM)
        # Calculate total induced dipoles
        for i, new_field in enumerate(new_fields_global):
            new_ind_dipoles[i, :] = np.einsum('ij, j', polarizabilities[i], np.add(new_field, fields[i]))
        new_residue = new_ind_dipoles - old_ind_dipoles
        residue_norm = np.linalg.norm(new_residue)
        max_residue_norm = np.max(np.abs(new_residue))
        # Check convergence
        if residue_norm < threshold and max_residue_norm < threshold:
            break
        # Save residue and dipoles
        if iteration == 1:
            residue = np.expand_dims(new_residue, axis=2)
            ind_dipoles = np.expand_dims(ind_dipoles, axis=2)
        else:
            residue = np.concatenate((residue, np.expand_dims(new_residue, axis=2)), axis=2)
        ind_dipoles = np.concatenate((ind_dipoles, np.expand_dims(new_ind_dipoles, axis=2)), axis=2)
        # Add to error matrix
        for i in range(0, iteration):
            error = np.sum(residue[:, :, i] * residue[:, :, iteration - 1])
            error_matrix_full[i, iteration - 1] = error
            error_matrix_full[iteration - 1, i] = error
        # Start diis at the assigned iteration
        if iteration >= init_diis:
            new_ind_dipoles = direct_inversion_iterative_subspace(ind_dipoles=ind_dipoles,
                                                                  error_matrix_full=error_matrix_full,
                                                                  max_diis=max_diis,
                                                                  iteration=iteration)
        old_ind_dipoles = copy.deepcopy(new_ind_dipoles)
    # Check maximum convergence
    if iteration > max_iterations:
        raise RuntimeError("Did not converge after the maximum number of iterations.")

    return old_ind_dipoles, iteration


def induced_dipoles_dcji(coordinates: np.ndarray,
                         polarizabilities: np.ndarray,
                         exclusions: list,
                         indices: np.ndarray,
                         fields: np.ndarray,
                         starting_guess: np.ndarray,
                         mic: bool,
                         box: np.ndarray,
                         threshold: float,
                         max_iterations: Optional[int] = 100,
                         k_cluster: Optional[int] = 5,
                         cluster_size_range: Optional[int] = -1,
                         comm: Optional[MPI.Comm] = None
                         ) -> Tuple[np.ndarray, int]:
    """Solves for dipoles that are induced in particle.Atoms with the element-based formula of the DCJI method.
    DCJI uses K-means clustering to divide and conquer the system. Clusters are solved by Cholesky factorization and
    the Jacobi method is used to solve between clusters.

    Args:
        coordinates: Array of coordinates for all Atoms.
        polarizabilities: Array of polarizabilities for all Atoms.
        exclusions: List of exclusions for all Atoms.
        indices: Array of atom indices.
        fields: Array of static fields on all Atoms.
        starting_guess: Array of induced dipoles used as the starting guess.
        mic: Boolean indicating whether induced dipoles are calculated using mic scaled coordinates.
        box: box dimensions as described by three vectors spanning the box.
        threshold: Convergence threshold for the residue norm between the (k+1)th and (k)th set of induced dipoles.
        max_iterations: Maximum number of iterations.
        k_cluster: The number of clusters to form in K-means clustering, used for divide and conquer methods.
        cluster_size_range: The range of cluster size deviations from the mean cluster size in number of atoms. -1
        allows cluster deviations of any size.
        comm: The MPI communicator.

    Returns:
        ind_dipoles: Array of induced dipoles.
        new_fields: Array of fields originating from the induced dipoles.
        iteration: Number of iterations it took to converge the induced dipoles to the threshold.
    """
    if not isinstance(coordinates, np.ndarray) or not isinstance(polarizabilities, np.ndarray) or \
            not isinstance(exclusions, list) or not isinstance(indices, np.ndarray) or \
            not isinstance(fields, np.ndarray) or not isinstance(starting_guess, np.ndarray):
        raise ValueError("Wrong input format.")
    if mic:
        engine.set_mic_coords_idcs_exlcs(coordinates, indices, exclusions, box)
    else:
        engine.set_coords_idcs_exlcs(coordinates, indices, exclusions)
    if comm is None:
        return induced_dipoles_dcji_serial(coordinates=coordinates,
                                           polarizabilities=polarizabilities,
                                           exclusions=exclusions,
                                           indices=indices,
                                           fields=fields,
                                           starting_guess=starting_guess,
                                           mic=mic,
                                           threshold=threshold,
                                           max_iterations=max_iterations,
                                           k_cluster=k_cluster,
                                           cluster_size_range=cluster_size_range)
    else:
        return induced_dipoles_dcji_parallel(coordinates=coordinates,
                                             polarizabilities=polarizabilities,
                                             exclusions=exclusions,
                                             indices=indices,
                                             fields=fields,
                                             starting_guess=starting_guess,
                                             mic=mic,
                                             threshold=threshold,
                                             max_iterations=max_iterations,
                                             k_cluster=k_cluster,
                                             cluster_size_range=cluster_size_range,
                                             comm=comm)


def induced_dipoles_dcji_serial(coordinates: np.ndarray,
                                polarizabilities: np.ndarray,
                                exclusions: list,
                                indices: np.array,
                                fields: np.ndarray,
                                starting_guess: np.ndarray,
                                mic: bool,
                                threshold: float,
                                max_iterations: int,
                                k_cluster: int,
                                cluster_size_range: int
                                ) -> Tuple[np.ndarray, int]:
    """Solves for dipoles that are induced in particle.Atoms with the element-based formula of the DCJI method.
    DCJI uses K-means clustering to divide and conquer the system. Clusters are solved by Cholesky factorization and
    the Jacobi method is used to solve between clusters.

    Args:
        coordinates: Array of coordinates for all Atoms.
        polarizabilities: Array of polarizabilities for all Atoms.
        exclusions: List of exclusions for all Atoms.
        indices: Array of atom indices.
        fields: Array of static fields on all Atoms.
        starting_guess: Array of induced dipoles used as the starting guess.
        mic: Boolean indicating whether induced dipoles are calculated using mic scaled coordinates.
        threshold: Convergence threshold for the residue norm between the (k+1)th and (k)th set of induced dipoles.
        max_iterations: Maximum number of iterations.
        k_cluster: The number of clusters to form in K-means clustering, used for divide and conquer methods.
        cluster_size_range: The range of cluster size deviations from the mean cluster size in number of atoms. -1
        allows cluster deviations of any size.

    Returns:
        ind_dipoles: Array of induced dipoles.
        new_fields: Array of fields originating from the induced dipoles.
        iteration: Number of iterations it took to converge the induced dipoles to the threshold.
    """
    # Calculate induced dipoles from other induced dipoles
    old_ind_dipoles = starting_guess
    residue_norm = sys.float_info.max
    max_residue_norm = sys.float_info.max
    iteration = 0
    new_ind_dipoles = np.zeros([len(fields), 3], dtype=np.float64)
    # Identify clusters from k-means clustering
    clusters = kmeans_clustering(coordinates=coordinates,
                                 k_cluster=k_cluster,
                                 cluster_size_range=cluster_size_range)
    # Solve each cluster internally by Cholesky factorization
    decomp = divide_and_conquer(coordinates=coordinates,
                                polarizabilities=polarizabilities,
                                clusters=clusters,
                                exclusions=exclusions,
                                indices=indices,
                                mic=mic,
                                k_cluster=k_cluster)
    # Solve in between clusters with the Jacobi method
    while iteration <= max_iterations:
        iteration += 1
        engine.set_old_ind_dipoles(old_ind_dipoles)
        for k in range(0, k_cluster):
            cluster = np.where(clusters == k)[0]
            source = np.where(clusters != k)[0]
            new_fields = engine.target_source_ind_dipoles_fields(np.array(cluster, dtype=np.int64),
                                                                 np.array(source, dtype=np.int64), mic)
            # Calculate total induced dipoles
            new_ind_dipoles[cluster] = np.reshape(scipy.linalg.cho_solve(decomp[k],
                                                                         np.add(new_fields.ravel(),
                                                                                fields[cluster].ravel())),
                                                  (len(cluster), 3))
        new_residue = new_ind_dipoles - old_ind_dipoles
        residue_norm = np.linalg.norm(new_residue)
        max_residue_norm = np.max(np.abs(new_residue))
        # Check convergence
        if residue_norm < threshold and max_residue_norm < threshold:
            break
        old_ind_dipoles = copy.deepcopy(new_ind_dipoles)
    # Check maximum convergence
    if iteration > max_iterations:
        raise RuntimeError("Did not converge after the maximum number of iterations.")

    return old_ind_dipoles, iteration


def induced_dipoles_dcji_parallel(coordinates: np.ndarray,
                                  polarizabilities: np.ndarray,
                                  exclusions: list,
                                  indices: np.array,
                                  fields: np.ndarray,
                                  starting_guess: np.ndarray,
                                  mic: bool,
                                  threshold: float,
                                  max_iterations: int,
                                  k_cluster: int,
                                  cluster_size_range: int,
                                  comm: MPI.Comm = None
                                  ) -> Tuple[np.ndarray, int]:
    """Solves for dipoles that are induced in particle.Atoms with the element-based formula of the DCJI method.
    DCJI uses K-means clustering to divide and conquer the system. Clusters are solved by Cholesky factorization and
    the Jacobi method is used to solve between clusters.

    Args:
        coordinates: Array of coordinates for all Atoms.
        polarizabilities: Array of polarizabilities for all Atoms.
        exclusions: List of exclusions for all Atoms.
        indices: Array of atom indices.
        fields: Array of static fields on all Atoms.
        starting_guess: Array of induced dipoles used as the starting guess.
        mic: Boolean indicating whether induced dipoles are calculated using mic scaled coordinates.
        threshold: Convergence threshold for the residue norm between the (k+1)th and (k)th set of induced dipoles.
        max_iterations: Maximum number of iterations.
        k_cluster: The number of clusters to form in K-means clustering, used for divide and conquer methods.
        cluster_size_range: The range of cluster size deviations from the mean cluster size in number of atoms. -1
        allows cluster deviations of any size.
        comm: The MPI communicator.

    Returns:
        ind_dipoles: Array of induced dipoles.
        new_fields: Array of fields originating from the induced dipoles.
        iteration: Number of iterations it took to converge the induced dipoles to the threshold.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    avg, res = divmod(k_cluster, size)
    counts = [avg + 1 if p < res else avg for p in range(size)]
    start = sum(counts[:rank])
    end = sum(counts[:rank + 1])
    # Calculate induced dipoles from other induced dipoles
    old_ind_dipoles = starting_guess
    residue_norm = sys.float_info.max
    max_residue_norm = sys.float_info.max
    iteration = 0
    new_ind_dipoles = np.zeros([len(fields), 3], dtype=np.float64)
    new_ind_dipoles_local = np.zeros([len(fields), 3], dtype=np.float64)
    # Identify clusters from k-means clustering
    clusters = kmeans_clustering(coordinates=coordinates,
                                 k_cluster=k_cluster,
                                 cluster_size_range=cluster_size_range)
    # Solve each cluster internally by Cholesky factorization
    decomp = divide_and_conquer(coordinates=coordinates,
                                polarizabilities=polarizabilities,
                                clusters=clusters,
                                exclusions=exclusions,
                                indices=indices,
                                mic=mic,
                                k_cluster=k_cluster)
    # Solve in between clusters with the Jacobi method
    while iteration <= max_iterations:
        iteration += 1
        engine.set_old_ind_dipoles(old_ind_dipoles)
        for k in range(start, end):
            cluster = np.where(clusters == k)[0]
            source = np.where(clusters != k)[0]
            new_fields = engine.target_source_ind_dipoles_fields(np.array(cluster, dtype=np.int64),
                                                                 np.array(source, dtype=np.int64), mic)
            # Calculate total induced dipoles
            new_ind_dipoles_local[cluster] = np.reshape(scipy.linalg.cho_solve(decomp[k],
                                                                               np.add(new_fields.ravel(),
                                                                                      fields[cluster].ravel())),
                                                        (len(cluster), 3))
        comm.Allreduce(new_ind_dipoles_local, new_ind_dipoles, op=MPI.SUM)
        new_residue = new_ind_dipoles - old_ind_dipoles
        residue_norm = np.linalg.norm(new_residue)
        max_residue_norm = np.max(np.abs(new_residue))
        # Check convergence
        if residue_norm < threshold and max_residue_norm < threshold:
            break
        old_ind_dipoles = copy.deepcopy(new_ind_dipoles)
    # Check maximum convergence
    if iteration > max_iterations:
        raise RuntimeError("Did not converge after the maximum number of iterations.")

    return old_ind_dipoles, iteration


def induced_dipoles_dcjidiis(coordinates: np.ndarray,
                             polarizabilities: np.ndarray,
                             exclusions: list,
                             indices: np.ndarray,
                             fields: np.ndarray,
                             starting_guess: np.ndarray,
                             mic: bool,
                             box: np.ndarray,
                             threshold: float,
                             max_iterations: Optional[int] = 100,
                             init_diis: Optional[int] = 3,
                             max_diis: Optional[int] = 10,
                             k_cluster: Optional[int] = 5,
                             cluster_size_range: Optional[int] = -1,
                             comm: Optional[MPI.Comm] = None
                             ) -> Tuple[np.ndarray, int]:
    """Solves for dipoles that are induced in particle.Atoms with the element-based formula of the DCJIDIIS method.
    DCJIDIIS uses K-means clustering to divide and conquer the system. Clusters are solved by Cholesky factorization and
    the Jacobi method is used to solve between clusters with direct inversion in the iterative subspace to improve
    convergence.

    Args:
        coordinates: Array of coordinates for all Atoms.
        polarizabilities: Array of polarizabilities for all Atoms.
        exclusions: List of exclusions for all Atoms.
        indices: Array of atom indices.
        fields: Array of static fields on all Atoms.
        starting_guess: Array of induced dipoles used as the starting guess.
        mic: Boolean indicating whether induced dipoles are calculated using mic scaled coordinates.
        box: box dimensions as described by three vectors spanning the box.
        threshold: Convergence threshold for the residue norm between the (k+1)th and (k)th set of induced dipoles.
        max_iterations: Maximum number of iterations.
        max_diis: Maximum number of previous iterations to consider in the DIIS method.
        init_diis: Iteration number at which DIIS is initiated.
        k_cluster: The number of clusters to form in K-means clustering, used for divide and conquer methods.
        cluster_size_range: The range of cluster size deviations from the mean cluster size in number of atoms. -1
        allows cluster deviations of any size.
        comm: The MPI communicator.

    Returns:
        ind_dipoles: Array of induced dipoles.
        new_fields: Array of fields originating from the induced dipoles.
        iteration: Number of iterations it took to converge the induced dipoles to the threshold.
    """
    if not isinstance(coordinates, np.ndarray) or not isinstance(polarizabilities, np.ndarray) or \
            not isinstance(exclusions, list) or not isinstance(indices, np.ndarray) or \
            not isinstance(fields, np.ndarray) or not isinstance(starting_guess, np.ndarray):
        raise ValueError("Wrong input format.")
    if mic:
        engine.set_mic_coords_idcs_exlcs(coordinates, indices, exclusions, box)
    else:
        engine.set_coords_idcs_exlcs(coordinates, indices, exclusions)
    if comm is None:
        return induced_dipoles_dcjidiis_serial(coordinates=coordinates,
                                               polarizabilities=polarizabilities,
                                               exclusions=exclusions,
                                               indices=indices,
                                               fields=fields,
                                               starting_guess=starting_guess,
                                               mic=mic,
                                               threshold=threshold,
                                               max_iterations=max_iterations,
                                               init_diis=init_diis,
                                               max_diis=max_diis,
                                               k_cluster=k_cluster,
                                               cluster_size_range=cluster_size_range)
    else:
        return induced_dipoles_dcjidiis_parallel(coordinates=coordinates,
                                                 polarizabilities=polarizabilities,
                                                 exclusions=exclusions,
                                                 indices=indices,
                                                 fields=fields,
                                                 starting_guess=starting_guess,
                                                 mic=mic,
                                                 threshold=threshold,
                                                 max_iterations=max_iterations,
                                                 init_diis=init_diis,
                                                 max_diis=max_diis,
                                                 k_cluster=k_cluster,
                                                 cluster_size_range=cluster_size_range,
                                                 comm=comm)


def induced_dipoles_dcjidiis_serial(coordinates: np.ndarray,
                                    polarizabilities: np.ndarray,
                                    exclusions: list,
                                    indices: np.array,
                                    fields: np.ndarray,
                                    starting_guess: np.ndarray,
                                    mic: bool,
                                    threshold: float,
                                    max_iterations: int,
                                    init_diis: int,
                                    max_diis: int,
                                    k_cluster: int,
                                    cluster_size_range: int
                                    ) -> Tuple[np.ndarray, int]:
    """Solves for dipoles that are induced in particle.Atoms with the element-based formula of the DCJIDIIS method.
    DCJIDIIS uses K-means clustering to divide and conquer the system. Clusters are solved by Cholesky factorization and
    the Jacobi method is used to solve between clusters with direct inversion in the iterative subspace to improve
    convergence.

    Args:
        coordinates: Array of coordinates for all Atoms.
        polarizabilities: Array of polarizabilities for all Atoms.
        exclusions: List of exclusions for all Atoms.
        indices: Array of atom indices.
        fields: Array of static fields on all Atoms.
        starting_guess: Array of induced dipoles used as the starting guess.
        mic: Boolean indicating whether induced dipoles are calculated using mic scaled coordinates.
        threshold: Convergence threshold for the residue norm between the (k+1)th and (k)th set of induced dipoles.
        max_iterations: Maximum number of iterations.
        max_diis: Maximum number of previous iterations to consider in the DIIS method.
        init_diis: Iteration number at which DIIS is initiated.
        k_cluster: The number of clusters to form in K-means clustering, used for divide and conquer methods.
        cluster_size_range: The range of cluster size deviations from the mean cluster size in number of atoms. -1
        allows cluster deviations of any size.

    Returns:
        ind_dipoles: Array of induced dipoles.
        new_fields: Array of fields originating from the induced dipoles.
        iteration: Number of iterations it took to converge the induced dipoles to the threshold.
    """
    # Calculate induced dipoles from other induced dipoles
    old_ind_dipoles = starting_guess
    ind_dipoles = starting_guess
    residue = np.array([0], dtype=np.float64)
    residue_norm = sys.float_info.max
    max_residue_norm = sys.float_info.max
    iteration = 0
    new_ind_dipoles = np.zeros([len(fields), 3], dtype=np.float64)
    error_matrix_full = np.ones((max_iterations + 1, max_iterations + 1), dtype=np.float64)
    error_matrix_full = -error_matrix_full
    error_matrix_full = error_matrix_full + np.diag(np.ones(max_iterations + 1, dtype=np.float64))
    # Identify clusters from k-means clustering
    clusters = kmeans_clustering(coordinates=coordinates,
                                 k_cluster=k_cluster,
                                 cluster_size_range=cluster_size_range)
    # Solve each cluster internally by Cholesky factorization
    decomp = divide_and_conquer(coordinates=coordinates,
                                polarizabilities=polarizabilities,
                                clusters=clusters,
                                exclusions=exclusions,
                                indices=indices,
                                mic=mic,
                                k_cluster=k_cluster)
    # Solve in between clusters with the Jacobi method
    while iteration <= max_iterations:
        iteration += 1
        engine.set_old_ind_dipoles(old_ind_dipoles)
        for k in range(0, k_cluster):
            cluster = np.where(clusters == k)[0]
            source = np.where(clusters != k)[0]
            new_fields = engine.target_source_ind_dipoles_fields(np.array(cluster, dtype=np.int64),
                                                                 np.array(source, dtype=np.int64),
                                                                 mic)
            # Calculate total induced dipoles
            new_ind_dipoles[cluster] = np.reshape(scipy.linalg.cho_solve(decomp[k],
                                                                         np.add(new_fields.ravel(),
                                                                                fields[cluster].ravel())),
                                                  (len(cluster), 3))
        new_residue = new_ind_dipoles - old_ind_dipoles
        residue_norm = np.linalg.norm(new_residue)
        max_residue_norm = np.max(np.abs(new_residue))
        # Check convergence
        if residue_norm < threshold and max_residue_norm < threshold:
            break
        # save residue and dipoles
        if iteration == 1:
            residue = np.expand_dims(new_residue, axis=2)
            ind_dipoles = np.expand_dims(ind_dipoles, axis=2)
        else:
            residue = np.concatenate((residue, np.expand_dims(new_residue, axis=2)), axis=2)
        ind_dipoles = np.concatenate((ind_dipoles, np.expand_dims(new_ind_dipoles, axis=2)), axis=2)
        # Add to error matrix
        for i in range(0, iteration):
            error = np.sum(residue[:, :, i] * residue[:, :, iteration - 1])
            error_matrix_full[i, iteration - 1] = error
            error_matrix_full[iteration - 1, i] = error
        # Start diis at the assigned iteration
        if iteration >= init_diis:
            new_ind_dipoles = direct_inversion_iterative_subspace(ind_dipoles=ind_dipoles,
                                                                  error_matrix_full=error_matrix_full,
                                                                  max_diis=max_diis,
                                                                  iteration=iteration)
        old_ind_dipoles = copy.deepcopy(new_ind_dipoles)
    # Check maximum convergence
    if iteration > max_iterations:
        raise RuntimeError("Did not converge after the maximum number of iterations.")

    return old_ind_dipoles, iteration


def induced_dipoles_dcjidiis_parallel(coordinates: np.ndarray,
                                      polarizabilities: np.ndarray,
                                      exclusions: list,
                                      indices: np.array,
                                      fields: np.ndarray,
                                      starting_guess: np.ndarray,
                                      mic: bool,
                                      threshold: float,
                                      max_iterations: int,
                                      init_diis: int,
                                      max_diis: int,
                                      k_cluster: int,
                                      cluster_size_range: int,
                                      comm: MPI.Comm = None,
                                      ) -> Tuple[np.ndarray, int]:
    """Solves for dipoles that are induced in particle.Atoms with the element-based formula of the DCJIDIIS method.
    DCJIDIIS uses K-means clustering to divide and conquer the system. Clusters are solved by Cholesky factorization and
    the Jacobi method is used to solve between clusters with direct inversion in the iterative subspace to improve
    convergence.

    Args:
        coordinates: Array of coordinates for all Atoms.
        polarizabilities: Array of polarizabilities for all Atoms.
        exclusions: List of exclusions for all Atoms.
        indices: Array of atom indices.
        fields: Array of static fields on all Atoms.
        starting_guess: Array of induced dipoles used as the starting guess.
        mic: Boolean indicating whether induced dipoles are calculated using mic scaled coordinates.
        threshold: Convergence threshold for the residue norm between the (k+1)th and (k)th set of induced dipoles.
        max_iterations: Maximum number of iterations.
        max_diis: Maximum number of previous iterations to consider in the DIIS method.
        init_diis: Iteration number at which DIIS is initiated.
        k_cluster: The number of clusters to form in K-means clustering, used for divide and conquer methods.
        cluster_size_range: The range of cluster size deviations from the mean cluster size in number of atoms. -1
        allows cluster deviations of any size.
        comm: The MPI communicator.

    Returns:
        ind_dipoles: Array of induced dipoles.
        new_fields: Array of fields originating from the induced dipoles.
        iteration: Number of iterations it took to converge the induced dipoles to the threshold.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    avg, res = divmod(k_cluster, size)
    counts = [avg + 1 if p < res else avg for p in range(size)]
    start = sum(counts[:rank])
    end = sum(counts[:rank + 1])
    # Calculate induced dipoles from other induced dipoles
    old_ind_dipoles = starting_guess
    ind_dipoles = starting_guess
    residue_norm = sys.float_info.max
    max_residue_norm = sys.float_info.max
    iteration = 0
    new_ind_dipoles = np.zeros([len(fields), 3], dtype=np.float64)
    new_ind_dipoles_local = np.zeros([len(fields), 3], dtype=np.float64)
    error_matrix_full = np.ones((max_iterations + 1, max_iterations + 1), dtype=np.float64)
    error_matrix_full = -error_matrix_full
    error_matrix_full = error_matrix_full + np.diag(np.ones(max_iterations + 1, dtype=np.float64))
    # Identify clusters from k-means clustering
    clusters = kmeans_clustering(coordinates=coordinates,
                                 k_cluster=k_cluster,
                                 cluster_size_range=cluster_size_range)
    # Solve each cluster internally by Cholesky factorization
    decomp = divide_and_conquer(coordinates=coordinates,
                                polarizabilities=polarizabilities,
                                clusters=clusters,
                                exclusions=exclusions,
                                indices=indices,
                                mic=mic,
                                k_cluster=k_cluster)
    # Solve in between clusters with the Jacobi method
    while iteration <= max_iterations:
        iteration += 1
        engine.set_old_ind_dipoles(old_ind_dipoles)
        for k in range(start, end):
            cluster = np.where(clusters == k)[0]
            source = np.where(clusters != k)[0]
            new_fields = engine.target_source_ind_dipoles_fields(np.array(cluster, dtype=np.int64),
                                                                 np.array(source, dtype=np.int64), mic)
            # Calculate total induced dipoles
            new_ind_dipoles_local[cluster] = np.reshape(scipy.linalg.cho_solve(decomp[k],
                                                                               np.add(new_fields.ravel(),
                                                                                      fields[cluster].ravel())),
                                                        (len(cluster), 3))
        comm.Allreduce(new_ind_dipoles_local, new_ind_dipoles, op=MPI.SUM)
        new_residue = new_ind_dipoles - old_ind_dipoles
        residue_norm = np.linalg.norm(new_residue)
        max_residue_norm = np.max(np.abs(new_residue))
        # Check convergence
        if residue_norm < threshold and max_residue_norm < threshold:
            break
        # save residue and dipoles
        if iteration == 1:
            residue = np.expand_dims(new_residue, axis=2)
            ind_dipoles = np.expand_dims(ind_dipoles, axis=2)
        else:
            residue = np.concatenate((residue, np.expand_dims(new_residue, axis=2)), axis=2)
        ind_dipoles = np.concatenate((ind_dipoles, np.expand_dims(new_ind_dipoles, axis=2)), axis=2)
        # Add to error matrix
        for i in range(0, iteration):
            error = np.sum(residue[:, :, i] * residue[:, :, iteration - 1])
            error_matrix_full[i, iteration - 1] = error
            error_matrix_full[iteration - 1, i] = error
        # Start diis at the assigned iteration
        if iteration >= init_diis:
            new_ind_dipoles = direct_inversion_iterative_subspace(ind_dipoles=ind_dipoles,
                                                                  error_matrix_full=error_matrix_full,
                                                                  max_diis=max_diis,
                                                                  iteration=iteration)
        old_ind_dipoles = copy.deepcopy(new_ind_dipoles)
    # Check maximum convergence
    if iteration > max_iterations:
        raise RuntimeError("Did not converge after the maximum number of iterations.")

    return old_ind_dipoles, iteration


def direct_inversion_iterative_subspace(ind_dipoles, error_matrix_full, max_diis, iteration):
    """Finds new guess for induced dipoles to enter the next iteration in the Jacobi method.

    Args:
        ind_dipoles: Array of induced dipoles from each previous iteration of the Jacobi method
        error_matrix_full: Array of summarized residual errors from each previous iteration in the Jacobi method.
        max_diis: Maximum number of previous iterations to consider in the DIIS method.
        iteration: Current iteration in the Jacobi method.

    Returns:
        new_ind_dipoles: Array of induced dipoles.
    """
    try:
        new_ind_dipoles = np.zeros(np.shape(ind_dipoles[:, :, 0]), dtype=np.float64)
    except IndexError:
        new_ind_dipoles = np.zeros(np.shape(ind_dipoles[:, :]), dtype=np.float64)
    m = min(max_diis, iteration)
    n = max(0, iteration - max_diis)
    # extract section of error matrix
    error_matrix = error_matrix_full[n:n + m + 1, n:n + m + 1]
    b = np.zeros(np.shape(error_matrix)[1], dtype=np.float64)
    b[-1] = -1
    try:
        coeff = np.linalg.solve(error_matrix, b)
    except np.linalg.LinAlgError:
        raise ValueError("Singular Matrix.")
    for i in range(0, m):
        new_ind_dipoles += ind_dipoles[:, :, i + n + 1] * coeff[i]

    return new_ind_dipoles


def kmeans_clustering(coordinates: np.ndarray,
                      k_cluster: int,
                      cluster_size_range: int,
                      ) -> np.ndarray:
    """Identifies clusters of atoms using K-means clustering on atomic coordinates
       and adjusts the cluster assignments to ensure clusters are of equal size.

    Args:
       coordinates: Array of coordinates of all Atoms.
       k_cluster: Number of clusters to form.
       cluster_size_range: The allowable deviation from the mean cluster size.
                           -1 indicates no equalization,
                           0 indicates clusters should be as close to the mean size as possible.
    Returns:
       nearest_centroids: Array of integers indicating which cluster each atom belongs to.
    """
    if k_cluster < 2 or k_cluster > len(coordinates):
        raise ValueError("k_cluster must be larger than 1 and smaller than or equal to the number of atoms.")
    # Perform k-means clustering
    kmeans = sklearn.cluster.KMeans(n_clusters=k_cluster, random_state=42)
    kmeans.fit(coordinates)
    nearest_centroids = kmeans.labels_
    centroids = kmeans.cluster_centers_
    # single iteration size equilibration
    if cluster_size_range > -1:
        # find center of system
        center = np.array([0.5 * (np.max(centroids[:, 0]) + np.min(centroids[:, 0])),
                           0.5 * (np.max(centroids[:, 1]) + np.min(centroids[:, 1])),
                           0.5 * (np.max(centroids[:, 2]) + np.min(centroids[:, 2]))], dtype=np.float64)
        # sort clusters by furthest to center
        sorted_indices = np.argsort(-((centroids[:, 0] - center[0]) ** 2 + (centroids[:, 1] - center[1]) ** 2 + (
                centroids[:, 2] - center[2]) ** 2))
        cluster_mean = len(coordinates) / k_cluster
        for i in range(0, k_cluster):
            # if cluster is small - take atoms from nearby clusters
            if np.count_nonzero(nearest_centroids == sorted_indices[i]) < np.floor(cluster_mean) - cluster_size_range:
                distances = np.zeros(len(coordinates), dtype=np.float64)
                # calculate squared distances from cluster to atoms
                for j in range(0, 3):
                    distances += (coordinates[:, j] - centroids[sorted_indices[i]][j]) ** 2
                # set distances to atoms in the i'th cluster to max
                max_dist = np.max(distances) + 1
                distances[nearest_centroids == sorted_indices[i]] = max_dist
                # check whether the nearest atom is in a cluster with atoms to spare
                while np.count_nonzero(nearest_centroids == sorted_indices[i]) < np.floor(
                        cluster_mean) - cluster_size_range:
                    from_cluster = nearest_centroids[np.argmin(distances)]
                    if (np.count_nonzero(nearest_centroids == from_cluster) > np.floor(
                            cluster_mean) - cluster_size_range
                            or np.where(sorted_indices == from_cluster)[0] > [i]):
                        # move atom to new cluster
                        nearest_centroids[np.argmin(distances)] = sorted_indices[i]
                    distances[np.argmin(distances)] = max_dist
            # if cluster is big - move atoms to nearby clusters
            if np.count_nonzero(nearest_centroids == sorted_indices[i]) > np.ceil(cluster_mean) + cluster_size_range:
                cluster_atoms = np.where(nearest_centroids == sorted_indices[i])[0]
                # calculate squared distances for atoms in the i'th cluster to all other clusters
                distances = np.zeros([len(coordinates[nearest_centroids == sorted_indices[i]]), k_cluster],
                                     dtype=np.float64)
                for j in range(0, 3):
                    for k in range(0, k_cluster):
                        distances[:, k] += (coordinates[nearest_centroids == sorted_indices[i]][:, j] - centroids[
                            k, j]) ** 2
                max_dist = np.max(distances) + 1
                distances[:, sorted_indices[i]] = max_dist
                # check whether the new cluster has room to spare
                while np.count_nonzero(nearest_centroids == sorted_indices[i]) > np.ceil(
                        cluster_mean) + cluster_size_range:
                    min_index = np.argmin(distances)
                    to_atom = np.unravel_index(min_index, distances.shape)[0]
                    to_cluster = np.unravel_index(min_index, distances.shape)[1]
                    if (np.count_nonzero(nearest_centroids == to_cluster) < np.ceil(cluster_mean) + cluster_size_range
                            or np.where(sorted_indices == to_cluster)[0] > [i]):
                        # move atom to new cluster
                        nearest_centroids[cluster_atoms[to_atom]] = to_cluster
                    distances[np.unravel_index(min_index, distances.shape)] = max_dist
    return nearest_centroids


def divide_and_conquer(coordinates: np.array,
                       polarizabilities: np.array,
                       clusters: np.array,
                       exclusions: list,
                       indices: np.array,
                       mic: bool,
                       k_cluster: int) -> list:
    """Finds the solutions to the induced dipoles within each cluster using Cholesky factorization.

    Args:
        coordinates: Array of coordinates for all Atoms.
        polarizabilities: Array of polarizabilities for all Atoms.
        clusters: Array of integers indicating which cluster each atom belongs to.
        exclusions: List of exclusions for all Atoms.
        indices: Array of atom indices.
        mic: Boolean indicating whether induced dipoles are calculated using mic scaled coordinates.
        k_cluster: The number of clusters to form in K-means clustering, used for divide and conquer methods.

    Returns:
        decomp: List of Cholesky decomposition matrices, one matrix for each cluster.
    """
    # Define Cholesky decompositions for each cluster
    decomp = [0] * k_cluster
    for i in range(0, k_cluster):
        cluster = np.where(clusters == i)[0]
        cluster_exclusions = [exclusions[i] for i in cluster]
        cluster_coordinates = coordinates[cluster]
        z = np.zeros((3 * len(cluster_coordinates), 3 * len(cluster_coordinates)), dtype=np.float64)
        for j in range(0, len(cluster_coordinates)):
            try:
                z[3 * j:3 * j + 3, 3 * j:3 * j + 3] = np.linalg.inv(polarizabilities[cluster][j])
            except np.linalg.LinAlgError:
                raise ValueError("Singular Matrix.")
            for k in range(j + 1, len(cluster_coordinates)):
                if indices[cluster[k]] in cluster_exclusions[j]:
                    continue
                z[3 * j:3 * j + 3, 3 * k:3 * k + 3] = engine.compute_cluster_t_tensor(cluster[j],
                                                                                      cluster[k],
                                                                                      np.array([1, 1, 1, 1, False],
                                                                                               dtype=np.int64),
                                                                                      mic)
                z[3 * k:3 * k + 3, 3 * j:3 * j + 3] = z[3 * j:3 * j + 3, 3 * k:3 * k + 3]
        decomp[i] = scipy.linalg.cho_factor(z, lower=False)

    return decomp
