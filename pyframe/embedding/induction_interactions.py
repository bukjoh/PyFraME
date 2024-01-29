from __future__ import annotations

import numpy as np
import copy
from pyframe.embedding import (constants, interaction_tensor, subsystem, vlx_interface, tensor_tools)
from typing import Union, Tuple, Optional


# def compute_static_contributions(quantum_subsystem: subsystem.QuantumSubsystem,
#                                  classical_subsystem: Union[subsystem.ClassicalSubsystem, list]
#                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """Calculates the static fields that are generated from the quantum subsystem and the classical subsystem/s.
#
#     Returns:
#         Static fields from the multipoles and nuclei, coordinates, fragments, and polarizabilities.
#     """
#     classical_fragments = []
#     coordinates = []
#     multipole_fields = None
#
#     if isinstance(classical_subsystem, list):
#         for c_subsystem in classical_subsystem:
#             if hasattr(c_subsystem, 'classical_fragments'):
#                 for fragments in c_subsystem.classical_fragments:
#                     classical_fragments.append(fragments)
#                     for atom in fragments.atoms:
#                         coordinates.append(atom.coordinate)
#             coordinates = np.array(coordinates)
#             classical_fragments = np.array(classical_fragments)
#             # Add multipole contributions
#             multipole_fields = np.zeros([len(coordinates), 3])
#             k = 0
#             for i, fragment_i in enumerate(classical_fragments):
#                 for coordinate in fragment_i.coordinates:
#                     field_component = np.zeros(3)
#                     for j, fragment_j in enumerate(classical_fragments):
#                         if i == j:
#                             continue
#                         field_component += fragment_j.potential(coordinate=coordinate,
#                                                                 pot_derivative_order=1)
#                     multipole_fields[k, :] = field_component
#                     k += 1
#     else:
#         if hasattr(classical_subsystem, 'classical_fragments'):
#             for fragments in classical_subsystem.classical_fragments:
#                 classical_fragments.append(fragments)
#                 for atom in fragments.atoms:
#                     coordinates.append(atom.coordinate)
#         coordinates = np.array(coordinates)
#         classical_fragments = np.array(classical_fragments)
#         # Add multipole contributions
#         multipole_fields = np.zeros([len(coordinates), 3])
#         k = 0
#         for i, fragment_i in enumerate(classical_fragments):
#             for coordinate in fragment_i.coordinates:
#                 field_component = np.zeros(3)
#                 for j, fragment_j in enumerate(classical_fragments):
#                     if i == j:
#                         continue
#                     field_component += fragment_j.potential(coordinate=coordinate,
#                                                             pot_derivative_order=1)
#                 multipole_fields[k, :] = field_component
#                 k += 1
#     # Add nuclear contributions
#     nuclear_fields = np.zeros([len(coordinates), 3])
#     if hasattr(quantum_subsystem, 'nuclei'):
#         for i, coordinate in enumerate(coordinates):
#             field_component = np.zeros(3)
#             for nucleus in quantum_subsystem.nuclei:
#                 field_component += nucleus.potential(coordinate=coordinate,
#                                                      pot_derivative_order=1)
#             nuclear_fields[i, :] = field_component
#     # Calculate polarizability tensor list of atoms
#     polarizabilities = np.zeros([len(coordinates), 3, 3])
#     k = 0
#     for fragment in classical_fragments:
#         for atom in fragment.atoms:
#             polarizabilities[k, :, :] = tensor_tools.uncompress_symmetric_matrix(atom.polarizability[4:10])
#             k += 1
#     return coordinates, multipole_fields, nuclear_fields, polarizabilities, classical_fragments
#
#
# def compute_induced_dipoles(density: np.ndarray,
#                             integral_drv: vlx_interface.EmbeddingIntegralDriver,
#                             coordinates: np.ndarray,
#                             polarizabilities: np.ndarray,
#                             classical_fragments: np.ndarray,
#                             threshold,
#                             multipole_fields: Optional[np.ndarray] = 0,
#                             nuclear_fields: Optional[np.ndarray] = 0
#                             ) -> Tuple[np.ndarray, np.ndarray]:
#     """Calculates the induced dipoles iteratively.
#
#     Args:
#         density: Electron density.
#         integral_drv: Integral driver to calculate the field of the density.
#         multipole_fields: Fields of multipoles.
#         nuclear_fields: Fields of nuclei.
#         coordinates: 2D-array (N_{atom}x3) of coordinates of all particle.Atom objects in the classical subsystem/s.
#         polarizabilities: 3D-array (N_{atom}x3x3) of the polarizabilities of all particle.Atom objects in the classical
#         subsystem/s.
#         classical_fragments: 1D-array (N_{fragment}) of all fragment.ClassicalFragment objects in the classical
#         subsystem/s.
#         threshold: Threshold for convergence of the induced dipoles.
#
#     Returns:
#         Induced dipoles and the total Field vector.
#     """
#     residue_norm = 1.
#     electric_fields = integral_drv.electric_fields(coordinates=coordinates, density=density)
#     static_fields = multipole_fields + nuclear_fields + electric_fields
#     ind_dipoles = np.zeros([len(static_fields), 3])
#     # calculate induced dipoles from static fields
#     static_induced_dipoles = np.zeros([len(static_fields), 3])
#     for i, field in enumerate(static_fields):
#         static_induced_dipoles[i, :] = np.einsum('ij, j', polarizabilities[i], field)
#     # first guess for induced dipoles
#     old_ind_dipoles = static_induced_dipoles
#     # calculate induced dipoles from other induced dipoles
#     iteration = 0
#     while residue_norm > threshold:
#         iteration += 1
#         new_fields = np.zeros([len(old_ind_dipoles), 3])
#         k = 0
#         for i, fragment_i in enumerate(classical_fragments):
#             for coordinate_i in fragment_i.coordinates:
#                 field_component = np.zeros(3)
#                 m = 0
#                 for j, fragment_j in enumerate(classical_fragments):
#                     if i == j:
#                         m += len(fragment_j.atoms)
#                         continue
#                     for coordinate_j in fragment_j.coordinates:
#                         # changed template to potential rather than interaction! could be wrong though..
#                         field_component += np.einsum('ij, j', interaction_tensor.
#                                                      compute_t_tensor(r_a=coordinate_j,
#                                                                       r_b=coordinate_i,
#                                                                       rank_a=1,
#                                                                       rank_b=1,
#                                                                       start_rank_a=1,
#                                                                       start_rank_b=1,
#                                                                       tensor_template=constants.values.
#                                                                       potential_tensor_template).data,
#                                                      old_ind_dipoles[m])
#                         m += 1
#                 new_fields[k, :] = field_component
#                 k += 1
#         # calculate induced dipoles
#         for i, new_field in enumerate(new_fields):
#             ind_dipoles[i, :] = np.einsum('ij, j',
#                                           polarizabilities[i], np.add(new_field,
#                                                                       static_fields[i]))
#         residue_norm = np.abs(np.linalg.norm(ind_dipoles - old_ind_dipoles) / np.linalg.norm(old_ind_dipoles))
#         old_ind_dipoles = copy.deepcopy(ind_dipoles)
#     print("Induced Dipoles Converged after:", f"{iteration:>2d}", " iterations!")
#     return ind_dipoles, electric_fields
#
#
# def compute_induced_dipoles2(density: np.ndarray,
#                             integral_drv: vlx_interface.EmbeddingIntegralDriver,
#                             coordinates: np.ndarray,
#                             polarizabilities: np.ndarray,
#                             threshold,
#                             exclusion_list: list,
#                             multipole_fields: Optional[np.ndarray] = 0,
#                             nuclear_fields: Optional[np.ndarray] = 0
#                             ) -> Tuple[np.ndarray, np.ndarray]:
#     """Calculates the induced dipoles iteratively.
#
#     Args:
#         density: Electron density.
#         integral_drv: Integral driver to calculate the field of the density.
#         multipole_fields: Fields of multipoles.
#         nuclear_fields: Fields of nuclei.
#         coordinates: 2D-array (N_{atom}x3) of coordinates of all particle.Atom objects in the classical subsystem/s.
#         polarizabilities: 3D-array (N_{atom}x3x3) of the polarizabilities of all particle.Atom objects in the classical
#         subsystem/s.
#         exclusion_list
#         threshold: Threshold for convergence of the induced dipoles.
#         exlusion_list
#
#     Returns:
#         Induced dipoles and the total Field vector.
#     """
#     electric_fields = integral_drv.electric_fields(coordinates=coordinates, density=density)
#     static_fields = multipole_fields + nuclear_fields + electric_fields
#     starting_guess = np.array([[-0.03903718, 0., 0.], [0.03903719, 0.,0.]])
#     ind_dipoles = compute_ind_dipoles(static_fields, coordinates, polarizabilities, exclusion_list, threshold, starting_guess)
#     return ind_dipoles, electric_fields


# def compute_ind_dipoles(external_field: np.array,
#                         coordinates: np.array,
#                         polarizabilities: np.array,
#                         exclusion_list: list,
#                         threshold: float,
#                         starting_guess: Optional[np.ndarray] = None
#                         ) -> np.ndarray:
#     """ Calculates induced dipoles for a specific density
#  Args:
#         external_field: External electric field that induces dipoles.
#         coordinates: 2D-array (N_{atom}x3) of coordinates of all particle.Atom objects in the classical subsystem/s.
#         polarizabilities: 3D-array (N_{atom}x3x3) of the polarizabilities of all particle.Atom objects in the classical
#         subsystem/s.
#         exclusion_list: Array of tuples that defines which particles do not interact with each other.
#         threshold: Threshold for convergence of the induced dipoles.
#         starting_guess:
#
#     Returns:
#         Induced dipoles and the total Field vector.
#
#     """
#     if starting_guess is not None:
#         old_ind_dipoles = starting_guess
#     else:
#         static_induced_dipoles = np.zeros([len(external_field), 3])
#         for i, field in enumerate(external_field):
#             static_induced_dipoles[i, :] = np.einsum('ij, j', polarizabilities[i], field)
#         # first guess for induced dipoles
#         old_ind_dipoles = static_induced_dipoles
#     ind_dipoles = np.zeros([len(external_field), 3])
#     residue_norm = 1.
#     # calculate induced dipoles from other induced dipoles
#     iteration = 0
#     while residue_norm > threshold:
#         iteration += 1
#         new_fields = np.zeros([len(old_ind_dipoles), 3])
#         for i, coordinate_i in enumerate(coordinates):
#             field_component = np.zeros(3)
#             for j, coordinate_j in enumerate(coordinates):
#                 if j in exclusion_list[i]:
#                     continue
#                 # changed template to potential rather than interaction! could be wrong though..
#                 field_component += np.einsum('ij, j', interaction_tensor.
#                                              compute_t_tensor(r_a=coordinate_j,
#                                                               r_b=coordinate_i,
#                                                               rank_a=1,
#                                                               rank_b=1,
#                                                               start_rank_a=1,
#                                                               start_rank_b=1,
#                                                               tensor_template=constants.values.
#                                                               potential_tensor_template).data,
#                                              old_ind_dipoles[j])
#             new_fields[i, :] = field_component
#         # calculate induced dipoles
#         for i, new_field in enumerate(new_fields):
#             ind_dipoles[i, :] = np.einsum('ij, j',
#                                           polarizabilities[i], np.add(new_field,
#                                                                       external_field[i]))
#         residue_norm = np.abs(np.linalg.norm(ind_dipoles - old_ind_dipoles) / np.linalg.norm(old_ind_dipoles))
#         old_ind_dipoles = copy.deepcopy(ind_dipoles)
#     print("Induced Dipoles Converged after:", f"{iteration:>2d}", " iterations!")
#     return ind_dipoles


def compute_induction_interaction(induced_dipoles: np.ndarray,
                                  coordinates: np.ndarray,
                                  integral_drv: vlx_interface.EmbeddingIntegralDriver
                                  ) -> np.ndarray:
    """Calculates the induction fock matrix contribution.

    Args:
        induced_dipoles: Induced dipoles in the environment.
        coordinates: 2D-array (N_{atom}x3) of coordinates of all particle.Atom objects in the classical subsystem/s.
        integral_drv: Integral driver to calculate the field of the density contracted with the multipoles.

    Returns:
        Induction fock matrix contribution.
    """
    fock_matrix = integral_drv.multipole_field_integrals(dipoles=induced_dipoles,
                                                         coordinates=coordinates)
    return fock_matrix


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
