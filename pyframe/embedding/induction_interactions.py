from __future__ import annotations

import numpy as np
import copy
from typing import Any
from pyframe.embedding import perturbation_tools as pt
from pyframe.embedding import subsystem
from pyframe.embedding.pert_tuple_cache import rspCache, rspPert, rspPertTuple


def ind_fock_matrix_contributions(classical_subsystem: subsystem.ClassicalSubsystem,
                                  integral_driver: Any
                                  ) -> np.ndarray:
    """Calculates the induced Fock matrix contributions h_es (M*t) from a ClassicalSubsystem and the one-electron
    integrals.

    Returns:
        Induced Fock matrix contribution.
    """
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


def ind_fock_matrix_gradient_contributions(classical_subsystem: subsystem.ClassicalSubsystem,
                                           integral_driver: Any
                                           ) -> np.ndarray:
    return integral_driver.ind_fock_matrix_contributions_gradient(coordinates=classical_subsystem.coordinates,
                                                                  induced_dipoles=classical_subsystem.induced_dipoles.
                                                                  induced_dipoles)


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
    energy_gradient = np.zeros([len(total_field_gradients), 3], dtype=np.float64)
    for i, field_gradient in enumerate(total_field_gradients):
        for j in range(len(induced_dipoles)):
            energy_gradient[i, 0] -= (induced_dipoles[j, 0] * field_gradient[j, 0] +
                                      induced_dipoles[j, 1] * field_gradient[j, 1] +
                                      induced_dipoles[j, 2] * field_gradient[j, 2])
            energy_gradient[i, 1] -= (induced_dipoles[j, 0] * field_gradient[j, 1] +
                                      induced_dipoles[j, 1] * field_gradient[j, 3] +
                                      induced_dipoles[j, 2] * field_gradient[j, 4])
            energy_gradient[i, 2] -= (induced_dipoles[j, 0] * field_gradient[j, 2] +
                                      induced_dipoles[j, 1] * field_gradient[j, 4] +
                                      induced_dipoles[j, 2] * field_gradient[j, 5])
    return energy_gradient


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
