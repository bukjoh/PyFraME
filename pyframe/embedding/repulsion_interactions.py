from __future__ import annotations

import numpy as np

from pyframe.embedding import subsystem, engine


def compute_repulsion_interactions(quantum_subsystem: subsystem.QuantumSubsystem,
                                   classical_subsystem: subsystem.ClassicalSubsystem,
                                   perturbation_order: int = 0,
                                   method: str = 'LJ',
                                   combination_rule: str = 'Lorentz-Berthelot'
                                   ) -> float | np.ndarray:
    """Computes the repulsion (Pauli-Repulsion) potential or gradients of the Nuclei of a QuantumSubsystem interacting
    with a ClassicalSubsystem.

    Args:
        quantum_subsystem: QuantumSubsystem
        classical_subsystem: ClassicalSubsystem
        perturbation_order: Order of geometric perturbation of the nuclei.
        method: Flag to set the method to be used.
        combination_rule: Flag to set the combination rule to be used. Default is Lorentz-Berthelot.

    Returns:
        Repulsion potential or gradient.
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
            if perturbation_order == 0:
                if comm is None:
                    return engine.unperturbed_lj_repulsion(np.array([0, len(classical_subsystem.coordinates)],
                                                                    dtype=np.int64))
                else:
                    rank = comm.Get_rank()
                    size = comm.Get_size()
                    avg, res = divmod(len(classical_subsystem.coordinates), size)
                    counts = [avg + 1 if p < res else avg for p in range(size)]
                    start = sum(counts[:rank])
                    end = sum(counts[:rank + 1])
                    return engine.unperturbed_lj_repulsion(np.array([start, end], dtype=np.int64))
            elif perturbation_order == 1:
                if comm is None:
                    return engine.lj_repulsion_gradient(np.array([0, len(classical_subsystem.coordinates)],
                                                                    dtype=np.int64))
                else:
                    rank = comm.Get_rank()
                    size = comm.Get_size()
                    avg, res = divmod(len(classical_subsystem.coordinates), size)
                    counts = [avg + 1 if p < res else avg for p in range(size)]
                    start = sum(counts[:rank])
                    end = sum(counts[:rank + 1])
                    return engine.lj_repulsion_gradient(np.array([start, end], dtype=np.int64))
            else:
                raise NotImplementedError("Perturbation order > 1 has not been implemented yet.")
