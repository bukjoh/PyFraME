from __future__ import annotations

import numpy as np
import copy
from pyframe.embedding import (constants, electrostatic_interactions, subsystem, vlx_interface, tensor_tools)
from typing import Union, Tuple


def compute_static_contributions(quantum_subsystem: subsystem.QuantumSubsystem,
                                 classical_subsystem: Union[subsystem.ClassicalSubsystem, list]
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the static fields that are generated from the quantum subsystem and the classical subsystem/s.

    Returns:
        Static fields from the multipoles and nuclei, coordinates, fragments, and polarizabilities.
    """
    classical_fragments = []
    coordinates = []
    multipole_fields = None

    if isinstance(classical_subsystem, list):
        for c_subsystem in classical_subsystem:
            if hasattr(c_subsystem, 'classical_fragments'):
                for fragments in c_subsystem.classical_fragments:
                    classical_fragments.append(fragments)
                    for atom in fragments.atoms:
                        coordinates.append(atom.coordinate)
            coordinates = np.array(coordinates)
            classical_fragments = np.array(classical_fragments)
            # Add multipole contributions
            multipole_fields = np.zeros([len(coordinates), 3])
            k = 0
            for i, fragment_i in enumerate(classical_fragments):
                for coordinate in fragment_i.coordinates:
                    field_component = np.zeros(3)
                    for j, fragment_j in enumerate(classical_fragments):
                        if i == j:
                            continue
                        field_component += fragment_j.potential(coordinate=coordinate,
                                                                pot_derivative_order=1)
                    multipole_fields[k, :] = field_component
                    k += 1
    else:
        if hasattr(classical_subsystem, 'classical_fragments'):
            for fragments in classical_subsystem.classical_fragments:
                classical_fragments.append(fragments)
                for atom in fragments.atoms:
                    coordinates.append(atom.coordinate)
        coordinates = np.array(coordinates)
        classical_fragments = np.array(classical_fragments)
        # Add multipole contributions
        multipole_fields = np.zeros([len(coordinates), 3])
        k = 0
        for i, fragment_i in enumerate(classical_fragments):
            for coordinate in fragment_i.coordinates:
                field_component = np.zeros(3)
                for j, fragment_j in enumerate(classical_fragments):
                    if i == j:
                        continue
                    field_component += fragment_j.potential(coordinate=coordinate,
                                                            pot_derivative_order=1)
                multipole_fields[k, :] = field_component
                k += 1
    # Add nuclear contributions
    nuclear_fields = np.zeros([len(coordinates), 3])
    if hasattr(quantum_subsystem, 'nuclei'):
        for i, coordinate in enumerate(coordinates):
            field_component = np.zeros(3)
            for nucleus in quantum_subsystem.nuclei:
                field_component += nucleus.potential(coordinate=coordinate,
                                                     pot_derivative_order=1)
            nuclear_fields[i, :] = field_component
    # Calculate polarizability tensor list of atoms
    polarizabilities = np.zeros([len(coordinates), 3, 3])
    k = 0
    for fragment in classical_fragments:
        for atom in fragment.atoms:
            polarizabilities[k, :, :] = tensor_tools.uncompress_symmetric_matrix(atom.polarizability[4:10])
            k += 1
    return coordinates, multipole_fields, nuclear_fields, polarizabilities, classical_fragments


def compute_induced_dipoles(density: np.ndarray,
                            integral_drv: vlx_interface.EmbeddingIntegralDriver,
                            static_fields: np.ndarray,
                            coordinates: np.ndarray,
                            polarizabilities: np.ndarray,
                            classical_fragments: np.ndarray,
                            threshold
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the induced dipoles iteratively.

    Args:
        density: Electron density.
        integral_drv: Integral driver to calculate the field of the density.
        static_fields: Fields of the nuclei and multipoles.
        coordinates: 2D-array (N_{atom}x3) of coordinates of all particle.Atom objects in the classical subsystem/s.
        polarizabilities: 3D-array (N_{atom}x3x3) of the polarizabilities of all particle.Atom objects in the classical
        subsystem/s.
        classical_fragments: 1D-array (N_{fragment}) of all fragment.ClassicalFragment objects in the classical
        subsystem/s.
        threshold: Threshold for convergence of the induced dipoles.

    Returns:
        Induced dipoles and the total Field vector.
    """
    ind_dipoles = np.zeros([len(static_fields), 3])
    residue_norm = 1.
    # electric contribution to static fields
    electric_fields = integral_drv.electric_fields(coordinates=coordinates, density=density)
    total_field = np.add(static_fields, electric_fields)
    # calculate induced dipoles from static fields
    static_induced_dipoles = np.zeros([len(total_field), 3])
    for i, field in enumerate(total_field):
        static_induced_dipoles[i, :] = np.einsum('ij, j', polarizabilities[i], field)
    # first guess for induced dipoles
    old_ind_dipoles = static_induced_dipoles
    # calculate induced dipoles from other induced dipoles
    iteration = 0
    while residue_norm > threshold:
        iteration += 1
        new_fields = np.zeros([len(old_ind_dipoles), 3])
        k = 0
        for i, fragment_i in enumerate(classical_fragments):
            for coordinate_i in fragment_i.coordinates:
                field_component = np.zeros(3)
                m = 0
                for j, fragment_j in enumerate(classical_fragments):
                    if i == j:
                        m += len(fragment_j.atoms)
                        continue
                    for coordinate_j in fragment_j.coordinates:
                        # changed template to potential rather than interaction! could be wrong though..
                        field_component += np.einsum('ij, j', electrostatic_interactions.
                                                     compute_t_tensor(r_a=coordinate_j,
                                                                      r_b=coordinate_i,
                                                                      rank_a=1,
                                                                      rank_b=1,
                                                                      start_rank_a=1,
                                                                      start_rank_b=1,
                                                                      tensor_template=constants.values.
                                                                      potential_tensor_template).data,
                                                     old_ind_dipoles[m])
                        m += 1
                new_fields[k, :] = field_component
                k += 1
        # calculate induced dipoles
        for i, new_field in enumerate(new_fields):
            ind_dipoles[i, :] = np.einsum('ij, j',
                                          polarizabilities[i], np.add(new_field,
                                                                      total_field[i]))
        residue_norm = np.abs(np.linalg.norm(ind_dipoles - old_ind_dipoles) / np.linalg.norm(old_ind_dipoles))
        old_ind_dipoles = copy.deepcopy(ind_dipoles)
    print("Induced Dipoles Converged after:", f"{iteration:>2d}", " iterations!")
    return ind_dipoles, electric_fields


def compute_induction_interaction(induced_dipoles: np.ndarray,
                                  total_fields: np.ndarray,
                                  coordinates: np.ndarray,
                                  integral_drv: vlx_interface.EmbeddingIntegralDriver
                                  ) -> Tuple[float, np.ndarray]:
    """Calculates the induction energy and the fock matrix contributions.

    Args:
        induced_dipoles: Induced dipoles in the environment.
        total_fields: Fields from the electron density, nuclei, and multipoles.
        coordinates: 2D-array (N_{atom}x3) of coordinates of all particle.Atom objects in the classical subsystem/s.
        integral_drv: Integral driver to calculate the field of the density contracted with the multipoles.

    Returns:
        Induction energy and fock matrix contributions.
    """
    fock_matrix = integral_drv.multipole_field_integrals(dipoles=induced_dipoles,
                                                         coordinates=coordinates)
    return -0.5 * np.einsum('ij, ij', total_fields, induced_dipoles), fock_matrix


def compute_induction_energy(induced_dipoles: np.ndarray,
                             fields: np.ndarray) -> float:
    return -0.5 * np.einsum('ij, ij', fields, induced_dipoles)
