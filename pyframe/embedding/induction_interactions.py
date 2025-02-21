from __future__ import annotations

import numpy as np
from typing import Any
from pyframe.embedding import perturbation_tools as pt
from pyframe.embedding import subsystem
from pyframe.embedding.pert_tuple_cache import rspCache, rspPert, rspPertTuple


def ind_fock_matrix_contributions(classical_subsystem: subsystem.ClassicalSubsystem,
                                  integral_driver: Any
                                  ) -> np.ndarray:
    """Calculates the induced Fock matrix contributions f_ind (FB*t) from a ClassicalSubsystem and the one-electron
    integrals.

    Returns:
        Induced Fock matrix contribution.
    """
    # TODO check if works
    if classical_subsystem.comm is not None:
        avg, res = divmod(len(classical_subsystem.coordinates), classical_subsystem.comm.size)
        counts = [avg + 1 if p < res else avg for p in range(classical_subsystem.comm.size)]
        start = sum(counts[:classical_subsystem.comm.rank])
        end = sum(counts[:classical_subsystem.comm.rank + 1])
        ind_fock_matrix = integral_driver.induced_dipoles_potential_integrals(
            induced_dipoles=classical_subsystem.induced_dipoles.induced_dipoles[start:end],
            coordinates=classical_subsystem.coordinates[start:end])
        ind_fock_matrix = classical_subsystem.comm.allreduce(ind_fock_matrix)
        return ind_fock_matrix
    else:
        return integral_driver.induced_dipoles_potential_integrals(
            induced_dipoles=classical_subsystem.induced_dipoles.induced_dipoles,
            coordinates=classical_subsystem.coordinates)


def compute_induction_energy(induced_dipoles: np.ndarray,
                             total_fields: np.ndarray) -> float:
    """Calculates the induction energy contribution.

     Args:
        total_fields: Fields from the electron density, nuclei, and multipoles.
        induced_dipoles: Induced dipoles in the environment.

    Returns:
        Induction energy
    """
    return -0.5 * np.einsum('ij, ij', total_fields, induced_dipoles)


def induced_fock_matrix_contributions_gradient(classical_subsystem: subsystem.ClassicalSubsystem,
                                               integral_driver: Any
                                               ) -> np.ndarray:
    return integral_driver.induced_fock_matrix_contributions_gradient(multipole_coordinates=classical_subsystem.
                                                                      coordinates,
                                                                      induced_dipoles=classical_subsystem.
                                                                      induced_dipoles.induced_dipoles)


def compute_induction_energy_gradient(induced_dipoles: np.ndarray,
                                      total_field_gradients: np.ndarray) -> np.ndarray:
    """Calculates the induction energy contribution.

     Args:
        total_field_gradients: Field gradients from the electron density and nuclei.
        induced_dipoles: Induced dipoles in the environment.

    Returns:
        Induction energy gradient.
    """
    # TODO move into c++ layer?

    # sign has been changed from other branch -> internally I expect potentials + potential derivatives
    energy_gradient = np.zeros([len(total_field_gradients), 3], dtype=np.float64)
    for i, field_gradient in enumerate(total_field_gradients):
        for j in range(len(induced_dipoles)):
            energy_gradient[i, 0] += (induced_dipoles[j, 0] * field_gradient[j, 0] +
                                      induced_dipoles[j, 1] * field_gradient[j, 1] +
                                      induced_dipoles[j, 2] * field_gradient[j, 2])
            energy_gradient[i, 1] += (induced_dipoles[j, 0] * field_gradient[j, 1] +
                                      induced_dipoles[j, 1] * field_gradient[j, 3] +
                                      induced_dipoles[j, 2] * field_gradient[j, 4])
            energy_gradient[i, 2] += (induced_dipoles[j, 0] * field_gradient[j, 2] +
                                      induced_dipoles[j, 1] * field_gradient[j, 4] +
                                      induced_dipoles[j, 2] * field_gradient[j, 5])
    return energy_gradient


def compute_electronic_induction_energy_gradients(density_matrix: np.ndarray,
                                                  classical_subsystem: subsystem.ClassicalSubsystem,
                                                  integral_driver: Any) -> np.ndarray:
    """Calculates the electronic induction energy gradient (µ_ind * F_el)^g from a ClassicalSubsystem and
    the one-electron integrals gradients.

    Args:
        density_matrix: Density Matrix that is the source of the electronic field.
                Shape: (number of ao functions, number of ao functions)
                Dtype: np.float64
        classical_subsystem: ClassicalSubsystem object containing coordinates and induced dipoles.
        integral_driver: Integral driver that calculates the electronic field gradients on coordinates and contracts
        with the induced dipoles.

    Returns:
        Electronic induction energy gradient.
    """
    # TODO check if sign change here (since its potential based ind dipoles) makes sense.
    #  -> or general sign change maybe in the embedding state class to come in vlx
    return integral_driver.electronic_induction_energy_gradients(
        induced_dipoles=-1.0 * classical_subsystem.induced_dipoles.induced_dipoles,
        coordinates=classical_subsystem.coordinates,
        density_matrix=density_matrix)


def compute_induction_energy_hessian(density_matrix: np.ndarray,
                                     classical_subsystem: subsystem.ClassicalSubsystem,
                                     quantum_subsystem: subsystem.QuantumSubsystem,
                                     integral_driver: Any,
                                     threshold: float = 1e-8,
                                     max_iterations: int = 100,
                                     mic: bool = False,
                                     box: np.ndarray = np.array([]),
                                     solver: str = 'jacobi'
                                     ) -> np.ndarray:
    no_nuc = quantum_subsystem.num_nuclei
    hess_contr = np.zeros([3 * no_nuc, 3 * no_nuc])

    # Calculate F^gBF^g
    f_g = np.zeros([no_nuc, 3, classical_subsystem.num_atoms, 3])
    mu_g = np.zeros([no_nuc, 3, classical_subsystem.num_atoms, 3])
    # Calculate FBF^gg

    if classical_subsystem.comm is not None:
        rank = classical_subsystem.comm.rank
        size = classical_subsystem.comm.size
        # calculate f_el_g and f_nuc_g
        avg, res = divmod(len(classical_subsystem.coordinates), size)
        counts = [avg + 1 if p < res else avg for p in range(size)]
        start = sum(counts[:rank])
        end = sum(counts[:rank + 1])
        for idx in range(no_nuc):
            f_g[idx, :, start:end, :] += integral_driver.compute_electronic_field_gradients(
                coordinates=classical_subsystem.coordinates[start:end],
                density_matrix=density_matrix,
                i=idx
            )
        f_g = classical_subsystem.comm.allreduce(f_g)
        # Add nuclear Field contribution
        idx = np.array([[0, 1, 2],
                        [1, 3, 4],
                        [2, 4, 5]])
        f_g += np.swapaxes(np.take(quantum_subsystem.compute_nuclear_field_gradients(
            coordinates=classical_subsystem.coordinates), idx, axis=2), 1, 2)

        for idx in range(no_nuc):
            for k in range(3):
                mu_g = classical_subsystem.solve_perturbed_induced_dipoles(
                    threshold=threshold,
                    max_iterations=max_iterations,
                    mic=mic,
                    box=box,
                    solver=solver,
                    external_fields=f_g[idx, k])

        total_iterations = no_nuc * (no_nuc + 1) // 2
        iterations_per_process = total_iterations // size
        remainder = total_iterations % size
        start = rank * iterations_per_process + min(rank, remainder)
        end = start + iterations_per_process + (1 if rank < remainder else 0)
        for iteration in range(start, end):
            i, j = pt.iteration_to_pair(iteration, no_nuc)
            # Compute the 3x3 block for nuclei i and j
            hessian_block = np.einsum('aAb, cAb -> ac', f_g[i], mu_g[j])
            # Insert the 3x3 block into the correct position in hess_contr
            hess_contr[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] += hessian_block
            if i != j:
                hess_contr[3 * j: 3 * j + 3, 3 * i: 3 * i + 3] += hessian_block.T
        # Reduce the contributions from all processes
        hess_contr = classical_subsystem.comm.allreduce(hess_contr)
    else:
        for idx in range(no_nuc):
            f_g[idx] += integral_driver.compute_electronic_field_gradients(
                coordinates=classical_subsystem.coordinates,
                density_matrix=density_matrix,
                i=idx
            )
        # Add nuclear Field contribution
        idx = np.array([[0, 1, 2],
                        [1, 3, 4],
                        [2, 4, 5]])
        f_g += np.swapaxes(np.take(quantum_subsystem.compute_nuclear_field_gradients(
            coordinates=classical_subsystem.coordinates), idx, axis=2), 1, 2)
        for idx in range(no_nuc):
            for k in range(3):
                mu_g = classical_subsystem.solve_perturbed_induced_dipoles(
                    threshold=threshold,
                    max_iterations=max_iterations,
                    mic=mic,
                    box=box,
                    solver=solver,
                    external_fields=f_g[idx, k])
        for i in range(no_nuc):
            for j in range(no_nuc):
                if i > j:
                    continue  # Only compute for the upper triangle (i <= j)
                # Compute the 3x3 block for nuclei i and j
                hessian_block = np.einsum('aAb, cAb -> ac', f_g[i], mu_g[j])
                # Insert the block in the (i,j) position of the full Hessian
                hess_contr[3 * i:3 * i + 3, 3 * j:3 * j + 3] += hessian_block
                if i != j:
                    # By symmetry, the (j,i) block is the transpose.
                    hess_contr[3 * j:3 * j + 3, 3 * i:3 * i + 3] += hessian_block.T
        # Calculate FBF^gg
        # Add electronic contribution to the field Hessian

    # Add electronic µF^gg contribution
    hess_contr += integral_driver.compute_electronic_field_hessian(
        coordinates=classical_subsystem.coordinates,
        induced_dipoles=classical_subsystem.induced_dipoles.induced_dipoles,
        density_matrix=density_matrix)
    # Add nuclear µF^gg contributions
    mapping = {
        (0, 0, 0): 0,  # xxx
        (0, 0, 1): 1,  # xxy
        (0, 0, 2): 2,  # xxz
        (0, 1, 1): 3,  # xyy
        (0, 1, 2): 4,  # xyz
        (0, 2, 2): 5,  # xzz
        (1, 1, 1): 6,  # yyy
        (1, 1, 2): 7,  # yyz
        (1, 2, 2): 8,  # yzz
        (2, 2, 2): 9  # zzz
    }

    def expand_third_rank(ten_tensor):
        # ten_tensor has shape (num_nuc, num_coords, 10)
        num_nuc, num_coords, _ = ten_tensor.shape
        full_tensor = np.empty((num_nuc, num_coords, 3, 3, 3))

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Sort the indices to get the canonical ordering
                    key = tuple(sorted((i, j, k)))
                    pos = mapping[key]
                    full_tensor[:, :, i, j, k] = ten_tensor[:, :, pos]

        return full_tensor

    f_gg = expand_third_rank(quantum_subsystem.compute_nuclear_field_hessian(classical_subsystem.coordinates))
    contr_dip_f_gg = np.einsum('ncijk,ck->ncij', f_gg, classical_subsystem.induced_dipoles.induced_dipoles)
    h_blocks = contr_dip_f_gg.sum(axis=1)
    num_nuc = h_blocks.shape[0]
    nuc_hess_contr = np.zeros((3 * num_nuc, 3 * num_nuc))
    for i in range(num_nuc):
        nuc_hess_contr[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] = h_blocks[i]
    hess_contr += nuc_hess_contr
    return hess_contr


def compute_electronic_induction_fock_gradient(i: int,
                                               classical_subsystem: subsystem.ClassicalSubsystem,
                                               integral_driver: Any) -> np.ndarray:
    """Calculates the electronic induction energy gradient (µ_ind * F_el)^g from a ClassicalSubsystem and
    the one-electron integrals gradients.

    Args:
        i: Index of Nucleus "i".
        classical_subsystem: ClassicalSubsystem object containing coordinates and induced dipoles.
        integral_driver: Integral driver that calculates the one-electron integral gradients on coordinates and
        contracts them with the induced dipoles at those coordinates.

    Returns:
        Electronic induction Fock matrix gradient of Nucleus "i".
    """
    return integral_driver.electronic_induction_fock_gradient(
        induced_dipoles=-1.0 * classical_subsystem.induced_dipoles.induced_dipoles,
        coordinates=classical_subsystem.coordinates,
        i=i)


def compute_rsp_induction_energy(input_cache: rspCache,
                                 density_bank: dict,
                                 quantum_subsystem: subsystem.QuantumSubsystem,
                                 classical_subsystem: subsystem.ClassicalSubsystem,
                                 solver_threshold: float = 1e-8,
                                 solver_max_iter: int = 100,
                                 solver_name: str = 'jacobi'
                                 ) -> dict:
    # Fill up FBF cache and unique F cache without values
    fbf_cache = {}
    unique_f_cache = {}
    res_tuples = [[], []]
    pt.ind_contr_recurse(perts=input_cache.p_tuples[0],
                         res_tuples=res_tuples,
                         fbf_cache=fbf_cache,
                         unique_f_cache=unique_f_cache,
                         comps=input_cache.comps,
                         k=input_cache.k,
                         n=input_cache.n,
                         debug_print=False)

    # Fill up unique F_el cache without values
    f_el_cache = {}
    for unique_f in list(unique_f_cache.values()):
        dummy_res_tuples = [[]]
        id_map = [[j.id for j in m.p] for m in unique_f.p_tuples]
        id_map.append([n for n in range(len(id_map[0]))])
        pt.rsp_field_recurse(perts=unique_f.p_tuples[0],
                             res_tuples=dummy_res_tuples,
                             comps=unique_f.comps,
                             cache=f_el_cache,
                             kn=[unique_f.k, unique_f.n],
                             id_map=id_map,
                             dryrun=True,
                             vals_to_set=None,
                             debug_print=False)
    # FIXME calc_pert_el_fields not finished -> integrals missing
    # Calculate and set the components of f_el_cache
    pt.calc_pert_el_fields(density_bank=density_bank,
                           f_el_cache=f_el_cache,
                           quantum_subsystem=quantum_subsystem,
                           coordinates=classical_subsystem.coordinates)

    # Non-dryrun of rsp_field_recurse to add F_el contributions to unique_f_cache
    for key in unique_f_cache.keys():
        # Skips F unperturbed
        if not unique_f_cache[key].p_tuples[0].p:
            continue
        dummy_res_tuples = [[]]
        id_map = [[j.id for j in m.p] for m in unique_f_cache[key].p_tuples]
        id_map.append([n for n in range(len(id_map[0]))])
        pt.rsp_field_recurse(perts=unique_f_cache[key].p_tuples[0],
                             res_tuples=dummy_res_tuples,
                             comps=unique_f_cache[key].comps,
                             cache=f_el_cache,
                             kn=[unique_f_cache[key].k, unique_f_cache[key].n],
                             id_map=id_map,
                             dryrun=False,
                             vals_to_set=unique_f_cache[key],
                             debug_print=False)

    # Add F_nuc contributions to unique_f_cache
    pt.calc_nuclei_fields(unique_f_cache=unique_f_cache,
                          quantum_subsystem=quantum_subsystem,
                          coordinates=classical_subsystem.coordinates)
    # Fill up FBF cache
    fbf_results = {}
    for key in fbf_cache.keys():
        # hash IDs
        ids = [tuple([j.id for j in i.p]) for i in fbf_cache[key].p_tuples]
        h = hash([ids[0], ids[1]])
        h_mirror = hash([ids[1], ids[0]])
        # Loop over necessary components
        for comp in fbf_cache[key].comps:
            # Check if in FBF results
            if h in fbf_results:
                # Search for components
                if comp in fbf_results[h]:
                    fbf_cache[key].vals[comp] = fbf_results[h][comp]
                else:
                    fbf_cache[key].vals[comp] = pt.calc_pert_ind_contr(ids=ids,
                                                                       comp=comp,
                                                                       unique_f_cache=unique_f_cache,
                                                                       fbf_p_tuples=fbf_cache[key].p_tuples,
                                                                       quantum_subsystem=quantum_subsystem,
                                                                       classical_subsystem=classical_subsystem,
                                                                       threshold=solver_threshold,
                                                                       max_iterations=solver_max_iter,
                                                                       solver=solver_name)
                    fbf_results[h][comp] = fbf_cache[key].vals[comp]

            elif h_mirror in fbf_results:
                # Search for components
                if comp in fbf_results[h_mirror]:
                    fbf_cache[key].vals[comp] = fbf_results[h_mirror][comp]
                else:
                    fbf_cache[key].vals[comp] = pt.calc_pert_ind_contr(ids=ids,
                                                                       comp=comp,
                                                                       unique_f_cache=unique_f_cache,
                                                                       fbf_p_tuples=fbf_cache[key].p_tuples,
                                                                       quantum_subsystem=quantum_subsystem,
                                                                       classical_subsystem=classical_subsystem,
                                                                       threshold=solver_threshold,
                                                                       max_iterations=solver_max_iter,
                                                                       solver=solver_name)
                    fbf_results[h][comp] = fbf_cache[key].vals[comp]

            else:
                fbf_results[h] = {}
                # Not in FBF results and add to FBF results
                fbf_cache[key].vals[comp] = pt.calc_pert_ind_contr(ids=ids,
                                                                   comp=comp,
                                                                   unique_f_cache=unique_f_cache,
                                                                   fbf_p_tuples=fbf_cache[
                                                                       key].p_tuples,
                                                                   quantum_subsystem=quantum_subsystem,
                                                                   classical_subsystem=classical_subsystem,
                                                                   threshold=solver_threshold,
                                                                   max_iterations=solver_max_iter,
                                                                   solver=solver_name)
                fbf_results[h][comp] = fbf_cache[key].vals[comp]
        fbf_cache[key].values_are_set = True

    # Sum and collect results
    res_tuples = [[], []]
    e_ind_contr = {}
    pt.ind_contr_recurse(perts=input_cache.p_tuples[0],
                         res_tuples=res_tuples,
                         fbf_cache=fbf_cache,
                         comps=input_cache.comps,
                         k=input_cache.k,
                         n=input_cache.n,
                         e_ind_contr=e_ind_contr,
                         dryrun=False,
                         debug_print=False)

    return e_ind_contr
