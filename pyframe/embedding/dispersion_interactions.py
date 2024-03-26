from __future__ import annotations

import numpy as np

from pyframe.embedding import subsystem, engine


def compute_dispersion_interactions(quantum_subsystem: subsystem.QuantumSubsystem,
                                    classical_subsystem: subsystem.ClassicalSubsystem,
                                    method: str = 'LJ',
                                    combination_rule: str = 'Lorentz-Berthelot',
                                    perturbation_order: int = 0
                                    ) -> float | np.ndarray:
    comm = classical_subsystem.comm
    if method == 'LJ':
        if combination_rule == 'Lorentz-Berthelot':
            if perturbation_order == 0:
                engine.set_atoms_nuclei_coordinates_lj_sigma_epsilon(classical_subsystem.disp_lj_sigma,
                                                                     classical_subsystem.disp_lj_epsilon,
                                                                     classical_subsystem.coordinates,
                                                                     quantum_subsystem.disp_lj_sigma,
                                                                     quantum_subsystem.disp_lj_epsilon,
                                                                     quantum_subsystem.coordinates)
                engine.set_combination_rule(combination_rule)
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
                    return engine.unperturbed_lj_dispersion(np.array([start, end], dtype=np.int64))
