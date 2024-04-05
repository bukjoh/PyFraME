from __future__ import annotations

import numpy as np
from typing import Any


def compute_induction_interaction(induced_dipoles: np.ndarray,
                                  coordinates: np.ndarray,
                                  integral_drv: Any
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
    # TODO move into c++ layer?
    return -0.5 * np.einsum('ij, ij', total_fields, induced_dipoles)


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
            # Move * -1 into -=
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
