from __future__ import annotations

import numpy as np

from pyframe.embedding import polytensor, particle, fragment, subsystem, engine
from typing import Union, Any, Tuple


def compute_particle_interactions(particle_1: particle, particle_2: particle):
    """Calculate the electrostatic interaction between two Particles.

    Args:
        particle_1: Particle 1.
        particle_2: Particle 2.
    Returns:
        Electrostatic interaction energy.
        """
    if not isinstance(particle_1, (particle.Atom, particle.Nucleus)) or \
            not isinstance(particle_2, (particle.Atom, particle.Nucleus)):
        raise TypeError("Arguments must be instances of Atom or Nucleus")
    if isinstance(particle_1, particle.Atom) and isinstance(particle_2, particle.Atom):
        return polytensor.FirstDegreePolytensor(rank=particle_1.multipole_order,
                                                tensor_data=particle_1.
                                                potential(coordinate=particle_2.coordinate,
                                                          coord_multipole_order=particle_2.multipole_order)). \
            dot_first_degree(polytensor.FirstDegreePolytensor.multiply_elementwise(particle_2.
                                                                                   multipoles_with_degeneracy,
                                                                                   particle_2.taylor_coefficients))
    if isinstance(particle_1, particle.Nucleus) and isinstance(particle_2, particle.Nucleus):
        return (particle_1.potential(coordinate=particle_2.coordinate) * particle_2.charge)[0]
    if isinstance(particle_1, particle.Atom) and isinstance(particle_2, particle.Nucleus):
        return (particle_1.potential(coordinate=particle_2.coordinate) * particle_2.charge)[0]
    if isinstance(particle_1, particle.Nucleus) and isinstance(particle_2, particle.Atom):
        return (particle_2.potential(coordinate=particle_1.coordinate) * particle_1.charge)[0]


def compute_fragment_interactions(c_fragment_1: fragment.ClassicalFragment,
                                  c_fragment_2: fragment.ClassicalFragment
                                  ) -> float:
    """Calculate the electrostatic interaction between two Classical fragments.

    Args:
        c_fragment_1: Classical fragment 1.
        c_fragment_2: Classical fragment 2.
    Returns:
        Electrostatic interaction energy.
    """
    if not isinstance(c_fragment_1, fragment.ClassicalFragment) or \
            not isinstance(c_fragment_2, fragment.ClassicalFragment):
        raise TypeError("Arguments must be instances of ClassicalFragment")
    electrostatic_energy = 0.0
    for atom_1 in c_fragment_1.atoms:
        for atom_2 in c_fragment_2.atoms:
            if atom_2.index in atom_1.exclusions:
                continue
            electrostatic_energy += compute_particle_interactions(atom_1, atom_2)
    return electrostatic_energy


def compute_fragment_particle_interactions(c_particle: particle,
                                           c_fragment: fragment.ClassicalFragment
                                           ) -> float:
    """Calculate the electrostatic interaction between a Classical fragment and a Particle.

    Returns:
        Electrostatic interaction energy between a Classical fragment and a particle from the perspective of particle.
    """
    if not isinstance(c_fragment, fragment.ClassicalFragment):
        raise TypeError("c_fragment must be an instance of ClassicalFragment")
    if isinstance(c_particle, particle.Atom):
        electrostatic_energy = 0
        for atoms in c_fragment.atoms:
            electrostatic_energy += compute_particle_interactions(c_particle, atoms)
        return electrostatic_energy

    if isinstance(c_particle, particle.Nucleus):
        return (c_fragment.potential(coordinate=c_particle.coordinate) * c_particle.charge)[0]


def compute_electrostatic_interaction(quantum_subsystem: subsystem.QuantumSubsystem,
                                      classical_subsystem: Union[subsystem.ClassicalSubsystem, list],
                                      integral_driver: Any
                                      ) -> Tuple[float, np.ndarray]:
    """Calculate the electrostatic interaction between a Quantum subsystem and one or several Classical subsystems.

    Returns:
        Electrostatic nuclear interaction energy and the electrostatic Fock matrix contribution.
    """
    if not isinstance(quantum_subsystem, subsystem.QuantumSubsystem):
        raise TypeError("quantum_subsystem must be an instance of QuantumSubsystem")
    fock_matrix = None
    if isinstance(classical_subsystem, list):
        nuclear_energy = 0
        for c_subsystem in classical_subsystem:
            # E_nuc_es
            if c_subsystem.comm is None:
                engine.set_multipoles_multipoles_order(c_subsystem.degenerate_multipoles_with_taylor_coefficients,
                                                       c_subsystem.multipole_orders)
                engine.set_coords_nuc_coords_charges(c_subsystem.coordinates,
                                                     quantum_subsystem.charges,
                                                     quantum_subsystem.coordinates)
                nuclear_energy = engine.e_nuc_es(np.array([0, len(c_subsystem.coordinates)], dtype=np.int64))
            else:
                engine.set_multipoles_multipoles_order(c_subsystem.degenerate_multipoles_with_taylor_coefficients,
                                                       c_subsystem.multipole_orders)
                engine.set_coords_nuc_coords_charges(c_subsystem.coordinates,
                                                     quantum_subsystem.charges,
                                                     quantum_subsystem.coordinates)
                avg, res = divmod(len(c_subsystem.coordinates), c_subsystem.size)
                counts = [avg + 1 if p < res else avg for p in range(c_subsystem.size)]
                start = sum(counts[:c_subsystem.rank])
                end = sum(counts[:c_subsystem.rank + 1])
                nuclear_energy = engine.e_nuc_es(np.array([start, end], dtype=np.int64))
            # F_el_es
            fock_matrix = es_fock_matrix_contributions(classical_subsystem=c_subsystem,
                                                       integral_driver=integral_driver)
    else:
        # E_nuc_es
        if classical_subsystem.comm is None:
            engine.set_multipoles_multipoles_order(classical_subsystem.degenerate_multipoles_with_taylor_coefficients,
                                                   classical_subsystem.multipole_orders)
            engine.set_coords_nuc_coords_charges(classical_subsystem.coordinates,
                                                 quantum_subsystem.charges,
                                                 quantum_subsystem.coordinates)
            nuclear_energy = engine.e_nuc_es(np.array([0, len(classical_subsystem.coordinates)], dtype=np.int64))
        else:
            engine.set_multipoles_multipoles_order(classical_subsystem.degenerate_multipoles_with_taylor_coefficients,
                                                   classical_subsystem.multipole_orders)
            engine.set_coords_nuc_coords_charges(classical_subsystem.coordinates,
                                                 quantum_subsystem.charges,
                                                 quantum_subsystem.coordinates)
            avg, res = divmod(len(classical_subsystem.coordinates), classical_subsystem.size)
            counts = [avg + 1 if p < res else avg for p in range(classical_subsystem.size)]
            start = sum(counts[:classical_subsystem.rank])
            end = sum(counts[:classical_subsystem.rank + 1])
            nuclear_energy = engine.e_nuc_es(np.array([start, end], dtype=np.int64))

        # F_el_es
        fock_matrix = es_fock_matrix_contributions(classical_subsystem=classical_subsystem,
                                                   integral_driver=integral_driver)
    return nuclear_energy, fock_matrix


def compute_electrostatic_nuclear_energy(quantum_subsystem: subsystem.QuantumSubsystem,
                                         classical_subsystem: subsystem.ClassicalSubsystem):
    if classical_subsystem.comm is None:
        engine.set_multipoles_multipoles_order(classical_subsystem.degenerate_multipoles_with_taylor_coefficients,
                                               classical_subsystem.multipole_orders)
        engine.set_coords_nuc_coords_charges(classical_subsystem.coordinates,
                                             quantum_subsystem.charges,
                                             quantum_subsystem.coordinates)
        nuclear_energy = engine.e_nuc_es(np.array([0, len(classical_subsystem.coordinates)], dtype=np.int64))
    else:
        engine.set_multipoles_multipoles_order(classical_subsystem.degenerate_multipoles_with_taylor_coefficients,
                                               classical_subsystem.multipole_orders)
        engine.set_coords_nuc_coords_charges(classical_subsystem.coordinates,
                                             quantum_subsystem.charges,
                                             quantum_subsystem.coordinates)
        avg, res = divmod(len(classical_subsystem.coordinates), classical_subsystem.size)
        counts = [avg + 1 if p < res else avg for p in range(classical_subsystem.size)]
        start = sum(counts[:classical_subsystem.rank])
        end = sum(counts[:classical_subsystem.rank + 1])
        nuclear_energy = engine.e_nuc_es(np.array([start, end], dtype=np.int64))
    return nuclear_energy


def compute_electrostatic_nuclear_gradients(quantum_subsystem: subsystem.QuantumSubsystem,
                                            classical_subsystem: subsystem.ClassicalSubsystem):
    """Calculate gradient of electrostatic nuclear interaction between a QuantumSubsystem and a ClassicalSubsystem.

    Returns:
        Electrostatic nuclear gradients.
            Shape: (Number of Nuclei, 3)
            Dtype: np.float64
    """
    if classical_subsystem.comm is None:
        engine.set_multipoles_multipoles_order(classical_subsystem.degenerate_multipoles_with_taylor_coefficients,
                                               classical_subsystem.multipole_orders)
        engine.set_coords_nuc_coords_charges(classical_subsystem.coordinates,
                                             quantum_subsystem.charges,
                                             quantum_subsystem.coordinates)
        nuclear_gradients = engine.e_nuc_es_gradients(
            np.array([0, len(classical_subsystem.coordinates)], dtype=np.int64))
    else:
        engine.set_multipoles_multipoles_order(classical_subsystem.degenerate_multipoles_with_taylor_coefficients,
                                               classical_subsystem.multipole_orders)
        engine.set_coords_nuc_coords_charges(classical_subsystem.coordinates,
                                             quantum_subsystem.charges,
                                             quantum_subsystem.coordinates)
        avg, res = divmod(len(classical_subsystem.coordinates), classical_subsystem.size)
        counts = [avg + 1 if p < res else avg for p in range(classical_subsystem.size)]
        start = sum(counts[:classical_subsystem.rank])
        end = sum(counts[:classical_subsystem.rank + 1])
        nuclear_gradients = engine.e_nuc_es_gradients(np.array([start, end], dtype=np.int64))
        # TODO maybe sign change since its a gradient?
    return nuclear_gradients


def es_fock_matrix_contributions(classical_subsystem: subsystem.ClassicalSubsystem,
                                 integral_driver: Any
                                 ) -> np.ndarray:
    """Calculate the electrostatic Fock matrix contributions h_es (M*t) from a Classical subsystem and the one-electron
    integrals.

    Returns:
        Electrostatic Fock matrix contribution.
    """
    return integral_driver.multipole_potential_integrals(multipole_coordinates=classical_subsystem.coordinates,
                                                         multipole_orders=classical_subsystem.multipole_orders,
                                                         multipoles=classical_subsystem.
                                                         degenerate_multipoles_with_taylor_coefficients)


def es_fock_matrix_gradient_contributions(classical_subsystem: subsystem.ClassicalSubsystem,
                                          integral_driver: Any
                                          ) -> np.ndarray:
    """Calculate the gradient of the electrostatic Fock matrix contributions h_es (M*t) from a Classical subsystem and
    the one-electron integrals.

    Returns:
        Gradient of electrostatic Fock matrix contribution.
    """
    return integral_driver.multipole_potential_gradient_integrals(multipole_coordinates=classical_subsystem.coordinates,
                                                                  multipole_orders=classical_subsystem.multipole_orders,
                                                                  multipoles=classical_subsystem.
                                                                  degenerate_multipoles_with_taylor_coefficients)


def compute_perturbed_electrostatic_interaction(quantum_subsystem: subsystem.QuantumSubsystem,
                                                classical_subsystem: Union[subsystem.ClassicalSubsystem, list],
                                                perturbation_indices: list,
                                                nucleus_idx: int
                                                ) -> float:
    # nucleus index!
    # perturbed E_es_nuc
    # E_nuc_es
    # FIXME has to be tested
    if classical_subsystem.comm is None:
        engine.set_multipoles_multipoles_order(classical_subsystem.degenerate_multipoles_with_taylor_coefficients,
                                               classical_subsystem.multipole_orders)
        engine.set_coords_nuc_coords_charges(classical_subsystem.coordinates,
                                             quantum_subsystem.charges,
                                             quantum_subsystem.coordinates)
        nuclear_energy = engine.e_nuc_es_perturbed(np.array([0, len(classical_subsystem.coordinates),
                                                             nucleus_idx], dtype=np.int64),
                                                   [np.array([0, 0, 0], dtype=np.int64),
                                                    np.array(perturbation_indices, dtype=np.int64)])
    else:
        engine.set_multipoles_multipoles_order(classical_subsystem.degenerate_multipoles_with_taylor_coefficients,
                                               classical_subsystem.multipole_orders)
        engine.set_coords_nuc_coords_charges(classical_subsystem.coordinates,
                                             quantum_subsystem.charges,
                                             quantum_subsystem.coordinates)
        avg, res = divmod(len(classical_subsystem.coordinates), classical_subsystem.size)
        counts = [avg + 1 if p < res else avg for p in range(classical_subsystem.size)]
        start = sum(counts[:classical_subsystem.rank])
        end = sum(counts[:classical_subsystem.rank + 1])
        nuclear_energy = engine.e_nuc_es_perturbed(np.array([start, end, nucleus_idx], dtype=np.int64),
                                                   [np.array([0, 0, 0], dtype=np.int64),
                                                    np.array(perturbation_indices, dtype=np.int64)])
    # FIXME the electric contributions are missing
    return nuclear_energy
