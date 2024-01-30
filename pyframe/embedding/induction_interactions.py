from __future__ import annotations

import numpy as np
from pyframe.embedding import vlx_interface


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
