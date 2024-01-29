from __future__ import annotations

import numpy as np
import copy
from typing import Optional
from pyframe.embedding import density_matrix, tensor_tools, constants, interaction_tensor, vlx_interface


class Subsystem:
    """A Subsystem represents a subsystem of the whole system by partitioning the system through the multiscale modeling
    approach.
    """

    def __init__(self,
                 name: Optional[str]):
        self._name = name


# TODO
# some kind of self energy function? MM/MM energies? thats multipole - multipole
# how to do the QM/QM energy? thats like veloxchem total energy of the core sys
# + QM/MM energies? -> nuclei - multipole, density - multipole energy


class QuantumSubsystem(Subsystem):
    """A QuantumSubsystem represents a collection of QuantumFragments, Nuclei and DensityMatrices.

    Args:
        input_data: Filepath to JSON file that contains the input data.
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

        # TODO incorporate potential of the density?


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
        input_data: Filepath to JSON file that contains the input data.
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
        self.polarizabilities = np.zeros([self.num_atoms, 3, 3])
        k = 0
        for fragments in self.classical_fragments:
            for atom in fragments.atoms:
                self.polarizabilities[k, :, :] = tensor_tools.uncompress_symmetric_matrix(atom.polarizability[4:10])
                self.coordinates[k, :] = atom.coordinate[:]
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
        self.inducing_fields = None
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
        print(external_fields)
        print(self.multipole_fields)
        static_fields = self.multipole_fields + external_fields
        # First guess for induced dipoles
        if np.all(self.induced_dipoles == 0):
            starting_guess = np.zeros([self.num_atoms, 3])
            for i, field in enumerate(static_fields):
                starting_guess[i, :] = np.einsum('ij, j', self.polarizabilities[i], field)
        else:
            starting_guess = self.induced_dipoles
        # Calculate induced dipoles from other induced dipoles
        old_ind_dipoles = starting_guess
        ind_dipoles = np.zeros([len(external_fields), 3])
        residue_norm = 1.
        iteration = 0
        while residue_norm > threshold:
            iteration += 1
            new_fields = np.zeros([len(old_ind_dipoles), 3])
            k = 0
            for fragment_i in self.classical_fragments:
                for i, atom_i in enumerate(fragment_i.atoms):
                    field_component = np.zeros(3)
                    for fragment_j in self.classical_fragments:
                        for j, atom_j in enumerate(fragment_j.atoms):
                            if atom_j.index in atom_i.exclusions:
                                continue
                            # Changed template to potential rather than interaction! could be wrong though..
                            field_component += np.einsum('ij, j', interaction_tensor.
                                                         compute_t_tensor(r_a=atom_j.coordinate,
                                                                          r_b=atom_i.coordinate,
                                                                          rank_a=1,
                                                                          rank_b=1,
                                                                          start_rank_a=1,
                                                                          start_rank_b=1,
                                                                          tensor_template=constants.values.
                                                                          potential_tensor_template).data,
                                                         old_ind_dipoles[j])
                    new_fields[k, :] = field_component
                    k += 1
            # Calculate total induced dipoles
            for i, new_field in enumerate(new_fields):
                ind_dipoles[i, :] = np.einsum('ij, j',
                                              self.polarizabilities[i], np.add(new_field,
                                                                          static_fields[i]))
            residue_norm = np.abs(np.linalg.norm(ind_dipoles - old_ind_dipoles) / np.linalg.norm(old_ind_dipoles))
            old_ind_dipoles = copy.deepcopy(ind_dipoles)
        print("Induced Dipoles Converged after:", f"{iteration:>2d}", " iterations!")
        print(ind_dipoles)
        self.induced_dipoles = ind_dipoles
        self.inducing_fields = static_fields + new_fields


class ContinuumSubsystem(Subsystem):
    """A ContinuumSubsystem represents a dielectric continuum.
    """
    pass
