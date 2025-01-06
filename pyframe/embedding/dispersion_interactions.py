from __future__ import annotations

import numpy as np

from pyframe.embedding import subsystem, engine, pert_tuple_cache, perturbation_tools, polytensor
from typing import Optional
from mpi4py import MPI


def compute_dispersion_interactions(quantum_subsystem: subsystem.QuantumSubsystem,
                                    classical_subsystem: subsystem.ClassicalSubsystem,
                                    perturbed: bool = False,
                                    method: str = 'LJ',
                                    combination_rule: str = 'Lorentz-Berthelot',
                                    perturbation_cache: Optional[pert_tuple_cache.rspCache] = None
                                    ) -> float | dict:
    """Computes the perturbed or unperturbed dispersion (London dispersion) potential of the Nuclei of a
    QuantumSubsystem interacting with a ClassicalSubsystem.

    Args:
        quantum_subsystem: QuantumSubsystem
        classical_subsystem: ClassicalSubsystem
        perturbed: Flag to indicate if contribution is for geometric perturbations of the nuclei.
        method: Flag to set the method to be used.
        combination_rule: Flag to set the combination rule to be used. Default is Lorentz-Berthelot.
        perturbation_cache: rspCache containing the perturbation tuple and components (perturbed has to be True).

    Returns:
        Dispersion potential or gradient.
    """
    comm = classical_subsystem.comm
    if method == 'LJ':
        engine.set_atoms_nuclei_coordinates_lj_sigma_epsilon(classical_subsystem.rep_lj_sigma,
                                                             classical_subsystem.rep_lj_epsilon,
                                                             classical_subsystem.coordinates,
                                                             quantum_subsystem.rep_lj_sigma,
                                                             quantum_subsystem.rep_lj_epsilon,
                                                             quantum_subsystem.coordinates)
        if combination_rule == 'Lorentz-Berthelot':
            engine.set_combination_rule(combination_rule)
            if not perturbed:
                if comm is None:
                    return engine.unperturbed_lj_dispersion(np.array([0, len(classical_subsystem.coordinates)],
                                                                     dtype=np.int64))
                else:
                    rank = comm.Get_rank()
                    size = comm.Get_size()
                    avg, res = divmod(len(classical_subsystem.coordinates), size)
                    counts = [avg + 1 if p < res else avg for p in range(size)]
                    start = sum(counts[:rank])
                    end = sum(counts[:rank + 1])
                    global_contr = 0
                    local_contr = engine.unperturbed_lj_dispersion(np.array([start, end], dtype=np.int64))
                    comm.Allreduce(local_contr, global_contr, op=MPI.SUM)
                    return global_contr
            else:
                # Set global factorials
                engine.set_factorials(polytensor.statics.factorials)
                if comm is None:
                    start = 0
                    end = len(classical_subsystem.coordinates)
                else:
                    rank = comm.Get_rank()
                    size = comm.Get_size()
                    avg, res = divmod(len(classical_subsystem.coordinates), size)
                    counts = [avg + 1 if p < res else avg for p in range(size)]
                    start = sum(counts[:rank])
                    end = sum(counts[:rank + 1])
                # Test if perturbation tuple contains EL
                dispersion_contr = {}
                for i, p_tuple in enumerate(perturbation_cache.p_tuples):
                    el = False
                    for k in p_tuple.p:
                        if k.o == 'EL':
                            el = True
                    if el:
                        if not p_tuple.h in dispersion_contr:
                            dispersion_contr[p_tuple.h] = {}
                        for comp in perturbation_cache.comps:
                            dispersion_contr[p_tuple.h][comp[i]] = 0.0
                    else:
                        if not p_tuple.h in dispersion_contr:
                            dispersion_contr[p_tuple.h] = {}
                        # Loop over Components
                        for comp in perturbation_cache.comps:

                            multi_indices_xyz = [[np.array([1, 0, 0], dtype=np.int64),
                                                  np.array([0, 0, 0], dtype=np.int64)],
                                                 [np.array([0, 1, 0], dtype=np.int64),
                                                  np.array([0, 0, 0], dtype=np.int64)],
                                                 [np.array([0, 0, 1], dtype=np.int64),
                                                  np.array([0, 0, 0], dtype=np.int64)]]
                            # nuc_index = None
                            pert_idx = None
                            k_dict = {}
                            k_list = []
                            same_nuc = True
                            for counter, k in enumerate(comp[i]):
                                # Single integer to describe index and multiindex
                                nuc_index = k // 3
                                if pert_idx is None:
                                    pert_idx = nuc_index
                                # Check if same nucleus
                                if nuc_index != pert_idx:
                                    same_nuc = False
                                    break
                                mod = k % 3
                                k_dict[counter] = multi_indices_xyz[mod]
                                k_list.append(counter)
                            if not same_nuc:
                                dispersion_contr[p_tuple.h][comp[i]] = 0.0
                                continue
                            else:
                                # It is always the same nucleus
                                k_partitions = perturbation_tools.subsets_of_list(k_list)
                                k_perturbations = perturbation_tools.replace_keys_with_values(dictionary=k_dict,
                                                                                              nested_list=k_partitions)
                                # check if comp already in dict
                                if not comp[i] in dispersion_contr[p_tuple.h]:
                                    # Pass k partitions and nucleus index to the function
                                    if comm is None:
                                        dispersion_contr[p_tuple.h][comp[i]] = engine.perturbed_lj_dispersion(
                                            np.array([start, end, nuc_index],
                                                     dtype=np.int64), k_perturbations)
                                    else:
                                        global_contr = 0.0
                                        local_contr = engine.perturbed_lj_dispersion(
                                            np.array([start, end, nuc_index], dtype=np.int64), k_perturbations)
                                        comm.Allreduce(local_contr, global_contr, op=MPI.SUM)
                                        dispersion_contr[p_tuple.h][comp[i]] = global_contr
                return dispersion_contr
        else:
            raise NotImplementedError("This combination rule has not been implemented yet.")
    else:
        raise NotImplementedError("This method has not been implemented yet.")


def compute_dispersion_interactions_gradient(quantum_subsystem: subsystem.QuantumSubsystem,
                                             classical_subsystem: subsystem.ClassicalSubsystem,
                                             method: str = 'LJ',
                                             combination_rule: str = 'Lorentz-Berthelot'
                                             ) -> np.ndarray:
    """Computes the dispersion (London dispersion) gradients of the Nuclei of a QuantumSubsystem interacting
    with a ClassicalSubsystem.

    Args:
        quantum_subsystem: QuantumSubsystem
        classical_subsystem: ClassicalSubsystem
        method: Flag to set the method to be used.
        combination_rule: Flag to set the combination rule to be used. Default is Lorentz-Berthelot.

    Returns:
        Dispersion gradient.
    """
    comm = classical_subsystem.comm
    if method == 'LJ':
        engine.set_atoms_nuclei_coordinates_lj_sigma_epsilon(classical_subsystem.rep_lj_sigma,
                                                             classical_subsystem.rep_lj_epsilon,
                                                             classical_subsystem.coordinates,
                                                             quantum_subsystem.rep_lj_sigma,
                                                             quantum_subsystem.rep_lj_epsilon,
                                                             quantum_subsystem.coordinates)
        if combination_rule == 'Lorentz-Berthelot':
            engine.set_combination_rule(combination_rule)
            if comm is None:
                return engine.lj_dispersion_gradient(np.array([0, len(classical_subsystem.coordinates)],
                                                              dtype=np.int64))
            else:
                rank = comm.Get_rank()
                size = comm.Get_size()
                avg, res = divmod(len(classical_subsystem.coordinates), size)
                counts = [avg + 1 if p < res else avg for p in range(size)]
                start = sum(counts[:rank])
                end = sum(counts[:rank + 1])
                global_gradient = np.zeros([quantum_subsystem.num_nuclei, 3], dtype=np.float64)
                local_gradient = engine.lj_dispersion_gradient(np.array([start, end], dtype=np.int64))
                comm.Allreduce(local_gradient, global_gradient, op=MPI.SUM)
                return global_gradient
        else:
            raise NotImplementedError("This combination rule has not been implemented yet.")
    else:
        raise NotImplementedError("This method has not been implemented yet.")
