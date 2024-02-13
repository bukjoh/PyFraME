from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Optional, Any
from pyframe.embedding import density_matrix, tensor_tools, solvers, electrostatic_interactions


class Subsystem:
    """A Subsystem represents a subsystem of the whole system by partitioning the system through the multiscale modeling
    approach.
    """

    def __init__(self,
                 name: Optional[str]):
        self.name = name


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
        if not nuclei:
            raise ValueError("QuantumSubsystem must have at least one Nucleus.")
        Subsystem.__init__(self, name=name)
        self.num_nuclei = len(nuclei)
        self.nuclei = nuclei
        self.density_matrix = dens_mat
        self.quantum_fragments = quantum_fragments
        self.coordinates = np.zeros([self.num_nuclei, 3])
        for i, nucleus in enumerate(nuclei):
            self.coordinates[i, :] = nucleus.coordinate[:]
    def static_potential(self,
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
            array_of_potentials: Parameter that indicates if the sum of all potentials and its derivatives of all nuclei
             is returned, or an array with the individual contributions.
        Returns:
            Electrostatic potential or its derivative of the fragment at coordinates. If coord_multipole_order is given,
            the derivatives with respect to the charge or multipole at coordinate are included.
        """
        pot = []
        for i, nucleus in enumerate(self.nuclei):
            pot.append(nucleus.potential(coordinate=coordinate,
                                         pot_derivative_order=pot_derivative_order,
                                         origin_derivative_order=origin_derivative_order,
                                         coord_multipole_order=coord_multipole_order))
        if array_of_potentials is False:
            return np.einsum('ij->j', np.array(pot))
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
                                coordinates: np.ndarray,
                                integral_drv: Any):
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
        if not classical_fragments:
            raise ValueError("ClassicalSubsystem must have at least one ClassicalFragment.")
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
                if len(atom.polarizability) == 10:
                    self.polarizabilities[k, :, :] = tensor_tools.uncompress_symmetric_matrix(atom.polarizability[4:10])
                self.coordinates[k, :] = atom.coordinate[:]
                self.exclusions.append(atom.exclusions)
                k += 1
        self.induced_dipoles = np.zeros([self.num_atoms, 3])
        self.multipole_fields = np.zeros([self.num_atoms, 3])
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
    def static_potential(self,
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
            return np.einsum('ij->j', np.array(pot))
        if array_of_potentials is True:
            return np.array(pot)


    def self_energy(self):
        return electrostatic_interactions.compute_classical_self_energy(self.classical_fragments)

    def solve_induced_dipoles(self,
                              threshold: float = 1e-10,
                              solver: Optional[str] = 'induced_dipoles_jacobi',
                              external_fields: Optional[np.ndarray] = None
                              ):
        if external_fields is not None:
            static_fields = self.multipole_fields + external_fields
        else:
            static_fields = self.multipole_fields
        # First guess for induced dipoles
        if np.all(self.induced_dipoles == 0):
            starting_guess = np.zeros([self.num_atoms, 3])
            for i, field in enumerate(static_fields):
                starting_guess[i, :] = np.einsum('ij, j', self.polarizabilities[i], field)
        else:
            residue_norm = np.abs(np.linalg.norm(external_fields - self.induced_dipoles.external_fields)
                                  / np.linalg.norm(self.induced_dipoles.external_fields))
            if residue_norm == 0:
                print("Residue norm between new and old external fields is 0, induced dipoles will not be "
                      "recalculated.")
                return
            elif residue_norm < 1e-6:
                print("Residue norm between new and old external fields is smaller than 1e-6, old induced dipoles will "
                      "be used as a starting guess.")
                starting_guess = self.induced_dipoles.induced_dipoles
            else:
                print("Residue norm between new and old external fields is larger than 1e-6, old induced dipoles will "
                      "not be used as a starting guess.")
                starting_guess = np.zeros([self.num_atoms, 3])
                for i, field in enumerate(static_fields):
                    starting_guess[i, :] = np.einsum('ij, j', self.polarizabilities[i], field)
        if solver == 'induced_dipoles_jacobi':
            induced_dipoles, induced_dipoles_fields, num_iter = (solvers.
                                                                 induced_dipoles_jacobi(coordinates=self.coordinates,
                                                                                        polarizabilities=self.
                                                                                        polarizabilities,
                                                                                        exclusions=self.exclusions,
                                                                                        indices=self.indices,
                                                                                        fields=static_fields,
                                                                                        starting_guess=starting_guess,
                                                                                        threshold=threshold))
        k = 0
        for fragment in self.classical_fragments:
            for atom in fragment.atoms:
                atom.induced_dipole = induced_dipoles[k]
                k += 1
        self.induced_dipoles = InducedDipoles(induced_dipoles=induced_dipoles,
                                              external_fields=external_fields,
                                              induced_dipole_fields=induced_dipoles_fields,
                                              number_of_iterations=num_iter,
                                              solver=solver)


@dataclass
class InducedDipoles(ClassicalSubsystem):
    """DataClass for induced dipoles. Contains array of induced dipoles, the external fields used to induce these
    dipoles, the field produced by the induced dipoles, the number of iterations it took to calculate the induced
    dipoles, and the name of the solver used.
    """
    induced_dipoles: np.ndarray
    external_fields: np.ndarray
    induced_dipole_fields: np.ndarray
    number_of_iterations: int
    solver: str


class ContinuumSubsystem(Subsystem):
    """A ContinuumSubsystem represents a dielectric continuum.
    """
    pass
