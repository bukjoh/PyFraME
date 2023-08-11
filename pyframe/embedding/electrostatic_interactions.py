from __future__ import annotations

import numpy as np
from pyframe.embedding import (polytensor, tensor_tools, particle, constants, fragment, subsystem, vlx_interface)
from typing import Optional, Union, Tuple


def compute_t_tensor(r_a: np.ndarray,
                     r_b: np.ndarray,
                     rank_a: int,
                     rank_b: int,
                     tensor_template: np.ndarray,
                     start_rank_b: Optional[int] = 0,
                     start_rank_a: Optional[int] = 0,
                     ) -> polytensor.SecondDegreePolytensor:
    """Builds and calculates the T tensor, which is a Matrix used to calculate the potential and interaction energy of
    Particles.

    Args:
        r_a: Cartesian coordinates of the first Particle.
        r_b: Cartesian coordinates of the second Particle.
        rank_a: Maximum column rank of the SecondDegreePolytensor.
        rank_b: Maximum row rank of the SecondDegreePolytensor.
        tensor_template: Template that contains the multi-indices to be calculated. Note that for calculating
        derivatives of the potential, and interaction energies the template should be different.
        start_rank_a: Minimum column rank of the SecondDegreePolytensor.
        start_rank_b: Minimum row rank of the SecondDegreePolytensor.

    Returns:
        T tensor as a SecondDegreePolytensor.
        (See Jon Applequist J. Math. Phys. 24, 736 (1983) for details on Polytensors.)
    """
    r_ab = r_b - r_a
    interaction_tensor = polytensor.SecondDegreePolytensor(rank_2=[start_rank_b, rank_b],
                                                           rank_1=[start_rank_a, rank_a])
    start_b = (start_rank_b - 1 + 1) * (start_rank_b - 1 + 2) * (start_rank_b - 1 + 3) // 6
    end_b = (rank_b + 1) * (rank_b + 2) * (rank_b + 3) // 6
    start_a = (start_rank_a - 1 + 1) * (start_rank_a - 1 + 2) * (start_rank_a - 1 + 3) // 6
    end_a = (rank_a + 1) * (rank_a + 2) * (rank_a + 3) // 6
    for i in range(start_a, end_a):
        for j in range(start_b, end_b):
            interaction_element = tensor_tools.compute_interaction_tensor_element(distance_vector=r_ab,
                                                                                  multi_index=tensor_template[i, j],
                                                                                  tensor_coefficients=constants.
                                                                                  values.tensor_coefficients)
            interaction_tensor.write_to_data(i=i - start_a, j=j - start_b, new_data=interaction_element)
    return interaction_tensor


def compute_atoms_interaction(atom_1: particle.Atom,
                              atom_2: particle.Atom
                              ) -> float:
    """Calculates the electrostatic interaction between two Atoms. The calculation is based on the vector-matrix-vector
    product M_1@T@M_2, where M are the compressed multipoles including the degeneracy factor, and T for the
    derivatives of 1/|r_{atom_2} - r_{atom_1}| wrt to the x, y, and z components.

    Returns:
        Electrostatic interaction energy.
    """
    return polytensor.FirstDegreePolytensor(rank=atom_1.multipole_order,
                                            tensor_data=atom_1.
                                            potential(coordinate=atom_2.coordinate,
                                                      coord_multipole_order=atom_2.multipole_order)). \
        dot_first_degree(polytensor.FirstDegreePolytensor.multiply_elementwise(atom_2.multipoles_with_degeneracy,
                                                                               atom_2.taylor_coefficients))


def compute_nuclei_interaction(nucleus_1: particle.Nucleus,
                               nucleus_2: particle.Nucleus
                               ) -> float:
    """Calculates the electrostatic interaction between two Nuclei.

    Returns:
        Electrostatic interaction energy.
    """
    return (nucleus_1.potential(coordinate=nucleus_2.coordinate) * nucleus_2.charge)[0]


def compute_atom_nucleus_interaction(atom: particle.Atom,
                                     nucleus: particle.Nucleus
                                     ) -> float:
    """Calculates the electrostatic interaction between an Atom and a Nucleus.

    Returns:
        Electrostatic interaction energy.
    """
    return (atom.potential(coordinate=nucleus.coordinate) * nucleus.charge)[0]


def compute_fragments_interaction(c_fragment_1: fragment.ClassicalFragment,
                                  c_fragment_2: fragment.ClassicalFragment
                                  ) -> float:
    """Calculates the electrostatic interaction between two Classical fragments.

    Returns:
        Electrostatic interaction energy.
    """
    electrostatic_energy = 0
    for atom_1 in c_fragment_1.atoms:
        for atom_2 in c_fragment_2.atoms:
            electrostatic_energy += compute_atoms_interaction(atom_1, atom_2)
    return electrostatic_energy


def compute_fragment_atom_interaction(atom: particle.Atom,
                                      c_fragment: fragment.ClassicalFragment
                                      ) -> float:
    """Calculates the electrostatic interaction between a Classical fragment and a particle.

    Returns:
        Electrostatic interaction energy between a Classical fragment and a particle from the perspective of particle1.
    """
    electrostatic_energy = 0
    for atoms in c_fragment.atoms:
        electrostatic_energy += compute_atoms_interaction(atom, atoms)
    return electrostatic_energy


def compute_fragment_nucleus_interaction(nucleus: particle.Nucleus,
                                         c_fragment: fragment.ClassicalFragment
                                         ) -> float:
    """Calculates the electrostatic interaction between a Classical fragment and a particle.

    Returns:
        Electrostatic interaction energy between a Classical fragment and a particle from the perspective of particle1.
    """
    return (c_fragment.potential(coordinate=nucleus.coordinate) * nucleus.charge)[0]


def compute_electrostatic_interaction(quantum_subsystem: subsystem.QuantumSubsystem,
                                      classical_subsystem: Union[subsystem.ClassicalSubsystem, list],
                                      integral_drv: vlx_interface.EmbeddingIntegralDriver
                                      ) -> Tuple[float, np.ndarray]:
    """Calculates the electrostatic interaction between a Quantum subsystem and one or several Classical subsystems.

    Returns:
        Electrostatic nuclear electrostatic interaction energy and the electric Fock matrix contribution.
    """
    fock_matrix = None
    if isinstance(classical_subsystem, list):
        nuclear_energy = 0
        for c_subsystem in classical_subsystem:
            # E_nuc_es
            for nucleus in quantum_subsystem.nuclei:
                nuclear_energy += (c_subsystem.potential(coordinate=nucleus.coordinate) * nucleus.charge)[0]
            # F_el_es
            fock_matrix = es_fock_matrix_contributions(classical_subsystem=c_subsystem,
                                                       integral_drv=integral_drv)
    else:
        nuclear_energy = 0
        # E_nuc_es
        for nucleus in quantum_subsystem.nuclei:
            nuclear_energy += (classical_subsystem.potential(coordinate=nucleus.coordinate) * nucleus.charge)[0]
        # F_el_es
        fock_matrix = es_fock_matrix_contributions(classical_subsystem=classical_subsystem,
                                                   integral_drv=integral_drv)
    return nuclear_energy, fock_matrix


def es_fock_matrix_contributions(classical_subsystem: subsystem.ClassicalSubsystem,
                                 integral_drv: vlx_interface.EmbeddingIntegralDriver):
    coordinates = []
    charges = []
    if hasattr(classical_subsystem, 'classical_fragments'):
        for frags in classical_subsystem.classical_fragments:
            for atom in frags.atoms:
                coordinates.append(atom.coordinate)
                charges.append(atom.multipoles_with_degeneracy.data[0] * atom.taylor_coefficients.data[0])
    if hasattr(classical_subsystem, 'atoms'):
        for atom in classical_subsystem.atoms:
            coordinates.append(atom.coordinate)
            charges.append(atom.multipoles_with_degeneracy.data[0] * atom.taylor_coefficients.data[0])
    fock_matrix_contribution = integral_drv.multipole_potential_integrals(charges=charges, coordinates=coordinates)
    return fock_matrix_contribution
