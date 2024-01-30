from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Optional
from pyframe.embedding import density_matrix, tensor_tools, vlx_interface, solvers


class Subsystem:
    """A Subsystem represents a subsystem of the whole system by partitioning the system through the multiscale modeling
    approach.
    """

    def __init__(self,
                 name: Optional[str]):
        self.name = name

# TODO incorporate potential of the density?
# TODO
# some kind of self energy function? MM/MM energies? thats multipole - multipole
# how to do the QM/QM energy? thats like veloxchem total energy of the core sys
# + QM/MM energies? -> nuclei - multipole, density - multipole energy


class QuantumSubsystem(Subsystem):
    """A QuantumSubsystem represents a collection of QuantumFragments, Nuclei and DensityMatrices.

    Args:
        nuclei: List of Nuclei.
        dens_mat: Density Matrix.
        quantum_fragments: Fragments of the QuantumSubsystem.
        name: Name of the QuantumSubsystem.
    """
    def __init__(self,
                 nuclei: list,
                 dens_mat: density_matrix.DensityMatrix,
                 quantum_fragments: Optional[list] = None,
                 name: Optional[str] = None,
                 ):
        Subsystem.__init__(self, name=name)
        self.nuclei = nuclei
        self.density_matrix = dens_mat
        self.quantum_fragments = quantum_fragments

    def potential(self,
                  coordinate: np.ndarray,
                  pot_derivative_order: Optional[int] = 0,
                  origin_derivative_order: Optional[int] = 0,
                  coord_multipole_order: Optional[int] = 0,
                  array_of_potentials: Optional[bool] = False
                  ) -> float | np.ndarray:
        """Calculates the sum of electrostatic potential and its derivatives of the atoms in a ClassicalFragment.

        Args:
            coordinate: Coordinates at which the potential is evaluated.
            pot_derivative_order: Order of the derivative of the potential.
            origin_derivative_order: Order of derivative with respect to the origin of the potential.
            coord_multipole_order: Multipole order at coordinate.
            array_of_potentials: Parameter that indicates if the sum of all potentials and its derivatives is returned,
            or an array with the individual contributions.
        Returns:
            Electrostatic potential or its derivative of the fragment at coordinates. If coord_multipole_order is given,
            the derivatives with respect to the charge or multipole at coordinate are included.
        """
        pot = []
        if hasattr(self, 'nuclei'):
            for nucleus in self.nuclei:
                pot.append(nucleus.potential(coordinate=coordinate,
                                             pot_derivative_order=pot_derivative_order,
                                             origin_derivative_order=origin_derivative_order,
                                             coord_multipole_order=coord_multipole_order))
        if array_of_potentials is False:
            return np.array(sum(pot))
        if array_of_potentials is True:
            return np.array(pot)


    def compute_nuclear_fields(self,
                               coordinates):
        nuclear_fields = np.zeros([len(coordinates), 3])
        for i, coordinate in enumerate(coordinates):
            field_component = np.zeros(3)
            for nucleus in self.nuclei:
                field_component += nucleus.potential(coordinate=coordinate,
                                                     pot_derivative_order=1)
            nuclear_fields[i, :] = field_component
        return nuclear_fields


    def compute_electric_fields(self,
                                coordinates,
                                integral_drv: vlx_interface.EmbeddingIntegralDriver):
        return integral_drv.electric_fields(coordinates=coordinates, density=self.density_matrix.density)


    def update_density(self, new_density: np.ndarray):
        """Updates the current density with a new density.
        """
        self.density_matrix.density = new_density


class ClassicalSubsystem(Subsystem):
    """A ClassicalSubsystem represents a collection of ClassicalFragments and/or Particles.

    Args:
        classical_fragments: Fragments of the ClassicalSubsystem.
        name: Name of the ClassicalSubsystem.
    """
    def __init__(self,
                 classical_fragments: list,
                 name: Optional[str] = None
                 ):
        Subsystem.__init__(self, name=name)
        self.classical_fragments = classical_fragments
        self.num_atoms = 0
        for fragments in self.classical_fragments:
            self.num_atoms += len(fragments.atoms)

        self.coordinates = np.zeros([self.num_atoms, 3])
        self.exclusions = []
        self.polarizabilities = np.zeros([self.num_atoms, 3, 3])
        self.indices = np.zeros(self.num_atoms)
        self.exclusions = []
        k = 0
        for fragments in self.classical_fragments:
            for atom in fragments.atoms:
                self.indices[k] = atom.index
                self.polarizabilities[k, :, :] = tensor_tools.uncompress_symmetric_matrix(atom.polarizability[4:10])
                self.coordinates[k, :] = atom.coordinate[:]
                self.exclusions.append(atom.exclusions)
                k += 1
        self.induced_dipoles = np.zeros([self.num_atoms, 3])
        self.multipole_fields = np.zeros([self.num_atoms, 3])
        self.induced_dipole_fields = None
        k = 0
        for fragment_i in self.classical_fragments:
            for i, atom_i in enumerate(fragment_i.atoms):
                field_component = np.zeros(3)
                for fragment_j in self.classical_fragments:
                    for j, atom_j in enumerate(fragment_j.atoms):
                        if atom_j.index in atom_i.exclusions:
                            continue
                        field_component += atom_j.potential(coordinate=atom_i.coordinate,
                                                            pot_derivative_order=1)
                self.multipole_fields[k, :] = field_component
                k += 1
    def potential(self,
                  coordinate: np.ndarray,
                  pot_derivative_order: Optional[int] = 0,
                  origin_derivative_order: Optional[int] = 0,
                  coord_multipole_order: Optional[int] = 0,
                  array_of_potentials: Optional[bool] = False
                  ) -> float | np.ndarray:
        """Calculates the sum of electrostatic potential and its derivatives of the atoms in a ClassicalFragment.

        Args:
            coordinate: Coordinates at which the potential is evaluated.
            pot_derivative_order: Order of the derivative of the potential.
            origin_derivative_order: Order of derivative with respect to the origin of the potential.
            coord_multipole_order: Multipole order at coordinate.
            array_of_potentials: Parameter that indicates if the sum of all potentials and its derivatives is returned,
            or an array with the individual contributions.
        Returns:
            Electrostatic potential or its derivative of the fragment at coordinates. If coord_multipole_order is given,
            the derivatives with respect to the charge or multipole at coordinate are included.
        """
        pot = []
        for fragments in self.classical_fragments:
            pot.append(fragments.potential(coordinate=coordinate,
                                           pot_derivative_order=pot_derivative_order,
                                           origin_derivative_order=origin_derivative_order,
                                           coord_multipole_order=coord_multipole_order))
        if array_of_potentials is False:
            return np.array(sum(pot))
        if array_of_potentials is True:
            return np.array(pot)


    def solve_induced_dipoles(self,
                              external_fields,
                              threshold):
        static_fields = self.multipole_fields + external_fields
        # First guess for induced dipoles
        if np.all(self.induced_dipoles == 0):
            starting_guess = np.zeros([self.num_atoms, 3])
            for i, field in enumerate(static_fields):
                starting_guess[i, :] = np.einsum('ij, j', self.polarizabilities[i], field)
        else:
            starting_guess = self.induced_dipoles.induced_dipoles

        induced_dipoles, induced_dipoles_fields = solvers.induced_dipoles_jacobi(coordinates=self.coordinates,
                                                                                 polarizabilities=self.polarizabilities,
                                                                                 exclusions=self.exclusions,
                                                                                 indices=self.indices,
                                                                                 fields=static_fields,
                                                                                 starting_guess=starting_guess,
                                                                                 threshold=threshold)
        k = 0
        for fragment in self.classical_fragments:
            for atom in fragment.atoms:
                atom.induced_dipole = induced_dipoles[k]
                k += 1
        self.induced_dipoles = InducedDipoles(induced_dipoles=induced_dipoles,
                                              external_fields=external_fields,
                                              induced_dipole_fields=induced_dipoles_fields)


@dataclass
class InducedDipoles(ClassicalSubsystem):
    """Class for keeping track of induced dipoles and the corresponding set of external fields.
    """
    induced_dipoles: np.ndarray
    external_fields: np.ndarray
    induced_dipole_fields: np.ndarray


class ContinuumSubsystem(Subsystem):
    """A ContinuumSubsystem represents a dielectric continuum.
    """
    pass
