from abc import ABC, abstractmethod

import numpy as np


class IntegralDriverTemplate(ABC):
    """Template for embedding driver that has to be supplied for PolarizableEmbedding. """

    @abstractmethod
    def electronic_fields(self,
                          coordinates: np.ndarray,
                          density_matrix: np.ndarray) -> np.ndarray:
        """Calculate the electronic fields on coordinates.

        Args:
            coordinates: Coordinates on which the fields are to be evaluated.
                Shape: (number of atoms, 3)
                Dtype: np.float64
            density_matrix: Density Matrix that is the source of the electronic field.
                Shape: (number of ao functions, number of ao functions)
                Dtype: np.float64

        Returns:
            Electronic fields. Shape: (number of atoms, 3) Dtype: np.float64.
        """
        raise NotImplementedError

    @abstractmethod
    def electronic_field_gradients(self,
                                   coordinates: np.ndarray,
                                   density_matrix: np.ndarray) -> np.ndarray:
        """Calculate the electronic fields on coordinates.

        Args:
            coordinates: Coordinates on which the fields are to be evaluated.
                Shape: (number of atoms, 3)
                Dtype: np.float64
            density_matrix: Density Matrix that is the source of the electronic field.
                Shape: (number of ao functions, number of ao functions)
                Dtype: np.float64

        Returns:
            Array of electronic field gradients in the same ordering as the input coordinates.
                Shape: (number of nuclei, number of atoms, 6) Dtype: np.float64.
        """
        raise NotImplementedError

    @abstractmethod
    def multipole_potential_integrals(self,
                                      multipole_coordinates: np.ndarray,
                                      multipole_orders: np.ndarray,
                                      multipoles: list[np.ndarray]) -> np.ndarray:
        """Calculate the electronic potential integrals and multiply with the multipoles.

        Args:
            multipole_coordinates: Coordinates of the Multipoles.
                Shape: (number of atoms, 3)
                Dtype: np.float64
            multipole_orders: Multipole orders of all multipoles.
                Shape: (number of atoms)
                Dtype: np.int64
            multipoles: Multipoles multiplied with degeneracy coefficients and taylor coefficients.
                Shape: (number of atoms, number of multipole elements)
                Dtype: np.float64

        Returns:
            Product of electronic potential integrals and multipoles.
                Shape: (number of ao functions, number of ao functions)
                Dtype: np.float64
        """
        raise NotImplementedError

    @abstractmethod
    def electronic_multipole_interaction_energy(self,
                                                multipole_coordinates: np.ndarray,
                                                multipole_orders: np.ndarray,
                                                multipoles: list[np.ndarray],
                                                density_matrix: np.ndarray) -> np.ndarray:
        """Calculate the interaction energy between multipoles and the electron density.

        Args:
            multipole_coordinates: Coordinates of the Multipoles.
                Shape: (number of atoms, 3)
                Dtype: np.float64
            multipole_orders: Multipole orders of all multipoles.
                Shape: (number of atoms)
                Dtype: np.int64
            multipoles: Multipoles multiplied with degeneracy coefficients and taylor coefficients.
                Shape: (number of atoms, number of multipole elements)
                Dtype: np.float64
            density_matrix: Density matrix that interacts with the multipoles.

        Returns:
            Interaction energy between multipoles and electrons.
                Shape: (1)
                Dtype: np.float64
        """
        raise NotImplementedError

    @abstractmethod
    def electronic_potential_integrals(self,
                                       coordinates: np.ndarray) -> np.ndarray:
        """Calculate the electronic potential integrals.

        Args:
            coordinates: Coordinates on which the integrals are to be evaluated.
                Shape (number of coordinates, 3)
                Dtype: np.float64

        Returns:
            Electronic potential integrals.
                Shape: (number of coordinates, number of ao functions, number of ao functions)
                Dtype: np.float64
        """
        raise NotImplementedError

    @abstractmethod
    def induced_dipoles_potential_integrals(self,
                                            induced_dipoles: np.ndarray,
                                            coordinates: np.ndarray) -> np.ndarray:
        """Calculate the electronic potential integrals and contract with the induced dipoles of Atoms.

        Args:
            induced_dipoles: Induced dipoles
                Shape (number of induced dipoles, 3)
                Dtype: np.float64
            coordinates: Coordinates of the induced dipoles on which the integrals are to be evaluated.
                Shape (number of induced dipoles, 3)
                Dtype: np.float64

        Returns:
            Product of the electronic potential integrals and the induced dipoles.
                Shape: (number of ao functions, number of ao functions)
                Dtype: np.float64
        """
        raise NotImplementedError

    @abstractmethod
    def induced_fock_matrix_contributions_gradient(self,
                                                   multipole_coordinates: np.ndarray,
                                                   induced_dipoles: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the induced Fock-matrix contributions.

        Args:
            multipole_coordinates: Coordinates of the Multipoles.
                Shape: (number of atoms, 3)
                Dtype: np.float64
            induced_dipoles: Induced dipoles on the Multipoles.
                Shape: (number of atoms, 3)
                Dtype: np.float64

        Returns:
            Gradient of induced Fock-matrix contributions.
                Shape: (number of nuclei, number of ao functions, number of ao functions)
                Dtype: np.float64
        """
        raise NotImplementedError

    @abstractmethod
    def multipole_potential_gradient_integrals(self,
                                               multipole_coordinates: np.ndarray,
                                               multipole_orders: np.ndarray,
                                               multipoles: list[np.ndarray]) -> np.ndarray:
        """Calculate the gradient of the electronic potential integrals and multiply with the multipoles.

        Args:
            multipole_coordinates: Coordinates of the Multipoles.
                Shape: (number of atoms, 3)
                Dtype: np.float64.
            multipole_orders: Multipole orders of all multipoles.
                Shape: (number of atoms)
                Dtype: np.int64
            multipoles: Multipoles multiplied with degeneracy coefficients and taylor coefficients.
                Shape: (number of atoms, number of multipole elements)
                Dtype: np.float64

        Returns:
            Product of gradient of electronic potential integrals and multipoles.
                Shape: (number of nuclei, number of ao functions, number of ao functions)
                Dtype: np.float64
        """
        raise NotImplementedError
