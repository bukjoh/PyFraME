from __future__ import annotations

import numpy as np

from pyframe.embedding import polytensor, particle, fragment, subsystem
from typing import Union, Any, Tuple
from mpi4py import MPI


def compute_particle_interactions(particle_1: particle, particle_2: particle):
    """Calculates the electrostatic interaction between two Particles.

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
    """Calculates the electrostatic interaction between two Classical fragments.

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
    """Calculates the electrostatic interaction between a Classical fragment and a Particle.

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
                                      integral_drv: Any
                                      ) -> Tuple[float, np.ndarray]:
    """Calculates the electrostatic interaction between a Quantum subsystem and one or several Classical subsystems.

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
            for nucleus in quantum_subsystem.nuclei:
                nuclear_energy += (c_subsystem.static_potential(coordinate=nucleus.coordinate) * nucleus.charge)[0]
            # F_el_es
            fock_matrix = es_fock_matrix_contributions(classical_subsystem=c_subsystem,
                                                       integral_drv=integral_drv)
    else:
        nuclear_energy = 0
        # E_nuc_es
        for nucleus in quantum_subsystem.nuclei:
            nuclear_energy += (classical_subsystem.static_potential(coordinate=nucleus.coordinate) * nucleus.charge)[0]
        # F_el_es
        fock_matrix = es_fock_matrix_contributions(classical_subsystem=classical_subsystem,
                                                   integral_drv=integral_drv)
    return nuclear_energy, fock_matrix


def es_fock_matrix_contributions(classical_subsystem: subsystem.ClassicalSubsystem,
                                 integral_drv: Any
                                 ) -> np.ndarray:
    """Calculates the electrostatic Fock matrix contributions h_es (M*t) from a Classical subsystem and the one-electron
    integrals.

    Returns:
        Electrostatic Fock matrix contribution.
    """
    # TODO check if integral driver also accepts np array and not list of np arrays.
    fock_matrix_contribution = integral_drv.multipole_potential_integrals(charges=classical_subsystem.charges,
                                                                          coordinates=classical_subsystem.coordinates)
    return fock_matrix_contribution
