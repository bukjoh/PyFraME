from __future__ import annotations

import numpy as np

from mpi4py import MPI
from dataclasses import dataclass
from typing import Optional, Any, Tuple
from pyframe.embedding import density_matrix, tensor_tools, solvers, electrostatic_interactions, engine, polytensor


class Subsystem:
    """A Subsystem represents a subsystem of the whole system by partitioning the system through the multiscale modeling
    approach.
    """

    def __init__(self,
                 name: Optional[str],
                 comm: Optional[MPI.Comm] = None
                 ):
        self.name = name
        self.comm = comm


class QuantumSubsystem(Subsystem):
    """A QuantumSubsystem represents a collection of QuantumFragments, Nuclei and DensityMatrices.

    Args:
        nuclei: List of Nuclei.
        dens_mat: Density Matrix.
        quantum_fragments: Fragments of the QuantumSubsystem.
        name: Name of the QuantumSubsystem.
        comm: The MPI communicator.
    """

    def __init__(self,
                 nuclei: list,
                 dens_mat: density_matrix.DensityMatrix,
                 quantum_fragments: Optional[list] = None,
                 name: Optional[str] = None,
                 comm: Optional[MPI.Comm] = None
                 ):
        if not nuclei:
            raise ValueError("QuantumSubsystem must have at least one Nucleus.")
        Subsystem.__init__(self, name=name, comm=comm)
        self.num_nuclei = len(nuclei)
        self.nuclei = nuclei
        self.density_matrix = dens_mat
        self.quantum_fragments = quantum_fragments
        self.coordinates = np.zeros([self.num_nuclei, 3], dtype=np.float64)
        self.charges = np.zeros([self.num_nuclei], dtype=np.float64)
        self._rep_lj_sigma = None
        self._rep_lj_epsilon = None
        self._disp_lj_sigma = None
        self._disp_lj_epsilon = None
        for i, nucleus in enumerate(self.nuclei):
            self.coordinates[i, :] = nucleus.coordinate[:]
            self.charges[i] = nucleus.charge[0]
        if self.comm is not None:
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()

    @property
    def rep_lj_sigma(self) -> np.ndarray:
        if getattr(self, '_rep_lj_sigma', None) is None:
            self._rep_lj_sigma = np.zeros([self.num_nuclei], dtype=np.float64)
            for i, nucleus in enumerate(self.nuclei):
                self._rep_lj_sigma[i] = nucleus.rep_parameters['lj_sigma']
        return self._rep_lj_sigma

    @property
    def rep_lj_epsilon(self) -> np.ndarray:
        if getattr(self, '_rep_lj_epsilon', None) is None:
            self._rep_lj_epsilon = np.zeros([self.num_nuclei], dtype=np.float64)
            for i, nucleus in enumerate(self.nuclei):
                self._rep_lj_epsilon[i] = nucleus.rep_parameters['lj_epsilon']
        return self._rep_lj_epsilon

    @property
    def disp_lj_sigma(self) -> np.ndarray:
        if getattr(self, '_disp_lj_sigma', None) is None:
            self._disp_lj_sigma = np.zeros([self.num_nuclei], dtype=np.float64)
            for i, nucleus in enumerate(self.nuclei):
                self._disp_lj_sigma[i] = nucleus.disp_parameters['lj_sigma']
        return self._disp_lj_sigma

    @property
    def disp_lj_epsilon(self) -> np.ndarray:
        if getattr(self, '_disp_lj_epsilon', None) is None:
            self._disp_lj_epsilon = np.zeros([self.num_nuclei], dtype=np.float64)
            for i, nucleus in enumerate(self.nuclei):
                self._disp_lj_epsilon[i] = nucleus.disp_parameters['lj_epsilon']
        return self._disp_lj_epsilon

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
                               coordinates
                               ) -> np.ndarray:
        """Calculates the electrostatic field from the nuclei.

        Args:
            coordinates: Array of coordinates on which the field is calculated for each set of coordinates.

        Returns:
            Array of nuclear fields on the different coordinates.
        """
        engine.set_coords_nuc_coords_charges(coordinates, self.charges, self.coordinates)
        if self.comm is not None:
            avg, res = divmod(len(coordinates), self.size)
            counts = [avg + 1 if p < res else avg for p in range(self.size)]
            start = sum(counts[:self.rank])
            end = sum(counts[:self.rank + 1])
            nuclear_fields_global = np.zeros([len(coordinates), 3])
            nuclear_fields_local = engine.nuclei_fields(np.array([start, end], dtype=np.int64))
            self.comm.Allreduce(nuclear_fields_local, nuclear_fields_global, op=MPI.SUM)
            return nuclear_fields_global

        else:
            return engine.nuclei_fields(np.array([0, len(coordinates)], dtype=np.int64))

    def compute_nuclear_field_gradients(self,
                                        coordinates
                                        ) -> np.ndarray:
        """Calculates the electrostatic field from the nuclei.

        Args:
            coordinates: Array of coordinates on which the field gradients are calculated for each set of coordinates.

        Returns:
            Array of nuclear field gradients on the different coordinates.
        """
        engine.set_coords_nuc_coords_charges(coordinates, self.charges, self.coordinates)
        if self.comm is not None:
            avg, res = divmod(len(coordinates), self.size)
            counts = [avg + 1 if p < res else avg for p in range(self.size)]
            start = sum(counts[:self.rank])
            end = sum(counts[:self.rank + 1])
            nuclear_fields_global = np.zeros([len(coordinates), 3])
            nuclear_fields_local = engine.nuclei_field_gradients(np.array([start, end], dtype=np.int64))
            self.comm.Allreduce(nuclear_fields_local, nuclear_fields_global, op=MPI.SUM)
            return nuclear_fields_global

        else:
            return engine.nuclei_field_gradients(np.array([0, len(coordinates)], dtype=np.int64))

    def compute_electric_fields(self,
                                coordinates: np.ndarray,
                                integral_drv: Any
                                ) -> np.ndarray:
        """Calculates the electrostatic field from the electron density.

        Args:
            coordinates: Array of coordinates on which the field is calculated for each set of coordinates.
            integral_drv: Integral driver to calculate the electric_fields of the electron density.

        Returns:
            Array of electric fields on the different coordinates.
        """
        return integral_drv.electric_fields(coordinates=coordinates, density=self.density_matrix.density)

    def update_density(self,
                       new_density: np.ndarray
                       ) -> None:
        """Updates the current density with a new density.
        """
        self.density_matrix.density = new_density


class ClassicalSubsystem(Subsystem):
    """A ClassicalSubsystem represents a collection of ClassicalFragments and/or Particles.

    Args:
        classical_fragments: Fragments of the ClassicalSubsystem.
        name: Name of the ClassicalSubsystem.
        comm: The MPI communicator.
    """

    def __init__(self,
                 classical_fragments: list,
                 name: Optional[str] = None,
                 comm: Optional[MPI.Comm] = None
                 ):
        if not classical_fragments:
            raise ValueError("ClassicalSubsystem must have at least one ClassicalFragment.")
        Subsystem.__init__(self, name=name, comm=comm)
        self.classical_fragments = classical_fragments
        self.num_atoms = 0
        for fragments in self.classical_fragments:
            self.num_atoms += len(fragments.atoms)
        self.coordinates = np.zeros([self.num_atoms, 3], dtype=np.float64)
        self.exclusions = []
        self.polarizabilities = np.zeros([self.num_atoms, 3, 3], dtype=np.float64)
        self.indices = np.zeros(self.num_atoms, dtype=np.int64)
        self.exclusions = []
        self.atoms = []
        self.charges = np.zeros([self.num_atoms], dtype=np.float64)
        self._rep_lj_sigma = None
        self._rep_lj_epsilon = None
        self._disp_lj_sigma = None
        self._disp_lj_epsilon = None
        self.multipoles_deg_taylor_coeff = []
        self.multipole_orders = np.zeros([self.num_atoms], dtype=np.int64)
        k = 0
        for fragments in self.classical_fragments:
            for atom in fragments.atoms:
                self.indices[k] = atom.index
                if len(atom.polarizability) == 10:
                    self.polarizabilities[k, :, :] = tensor_tools.uncompress_symmetric_matrix(atom.polarizability[4:10])
                self.coordinates[k, :] = atom.coordinate[:]
                self.exclusions.append(atom.exclusions)
                self.charges[k] = atom.multipoles.data[0]
                self.atoms.append(atom)
                self.multipoles_deg_taylor_coeff.append(polytensor.FirstDegreePolytensor.multiply_elementwise(
                    atom.multipoles_with_degeneracy,
                    atom.taylor_coefficients).data)
                self.multipole_orders[k] = atom.multipole_order
                k += 1
        self.induced_dipoles = InducedDipoles(induced_dipoles=np.zeros([self.num_atoms, 3]),
                                              external_fields=np.zeros([self.num_atoms, 3]),
                                              number_of_iterations=0,
                                              threshold=1e-8,
                                              solver="None")
        self.perturbed_induced_dipoles = []
        self._multipole_fields = None
        self._self_energy = None
        if self.comm is not None:
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()

    @property
    def rep_lj_sigma(self) -> np.ndarray:
        if getattr(self, '_rep_lj_sigma', None) is None:
            self._rep_lj_sigma = np.zeros([self.num_atoms], dtype=np.float64)
            for i, atom in enumerate(self.atoms):
                self._rep_lj_sigma[i] = atom.rep_parameters['lj_sigma']
        return self._rep_lj_sigma

    @property
    def rep_lj_epsilon(self) -> np.ndarray:
        if getattr(self, '_rep_lj_epsilon', None) is None:
            self._rep_lj_epsilon = np.zeros([self.num_atoms], dtype=np.float64)
            for i, atom in enumerate(self.atoms):
                self._rep_lj_epsilon[i] = atom.rep_parameters['lj_epsilon']
        return self._rep_lj_epsilon

    @property
    def disp_lj_sigma(self) -> np.ndarray:
        if getattr(self, '_disp_lj_sigma', None) is None:
            self._disp_lj_sigma = np.zeros([self.num_atoms], dtype=np.float64)
            for i, atom in enumerate(self.atoms):
                self._disp_lj_sigma[i] = atom.disp_parameters['lj_sigma']
        return self._disp_lj_sigma

    @property
    def disp_lj_epsilon(self) -> np.ndarray:
        if getattr(self, '_disp_lj_epsilon', None) is None:
            self._disp_lj_epsilon = np.zeros([self.num_atoms], dtype=np.float64)
            for i, atom in enumerate(self.atoms):
                self._disp_lj_epsilon[i] = atom.disp_parameters['lj_epsilon']
        return self._disp_lj_epsilon

    @property
    def multipole_fields(self):
        if getattr(self, '_multipole_fields', None) is None:
            self.compute_multipole_fields()
        return self._multipole_fields

    def compute_multipole_fields(self) -> None:
        """Computes the multipole fields from fragments.
        """
        engine.set_multipoles_multipoles_order(self.multipoles_deg_taylor_coeff,
                                               self.multipole_orders)
        engine.set_coords_idxs_exlcs(self.coordinates,
                                     self.indices,
                                     self.exclusions)
        self._multipole_fields = np.zeros([self.num_atoms, 3])
        if self.comm is not None:
            multipole_fields_local = np.zeros_like(self._multipole_fields)
            avg, res = divmod(len(self.coordinates), self.size)
            counts = [avg + 1 if p < res else avg for p in range(self.size)]
            start = sum(counts[:self.rank])
            end = sum(counts[:self.rank + 1])
            for i in range(start, end):
                multipole_fields_local[i, :] = engine.multipole_fields(np.array([i], dtype=np.int64)).T
            # Perform reduction
            self.comm.Allreduce([multipole_fields_local, MPI.DOUBLE],
                                [self._multipole_fields, MPI.DOUBLE],
                                op=MPI.SUM)
        else:
            for i in range(len(self.coordinates)):
                self._multipole_fields[i, :] = engine.multipole_fields(np.array([i], dtype=np.int64)).T

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
        # TODO has to be parallelized for multithreading
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

    @property
    def self_energy(self
                    ) -> float:
        """Calculates the electrostatic energy between all ClassicalFragments.

        Returns:
            Self energy of the ClassicalSubsystem.
        """
        if getattr(self, '_self_energy', None) is None:
            engine.set_multipoles_multipoles_order(self.multipoles_deg_taylor_coeff,
                                                   self.multipole_orders)
            engine.set_coords_idxs_exlcs(self.coordinates,
                                         self.indices,
                                         self.exclusions)
            total_iterations = (self.num_atoms - 1) * self.num_atoms // 2
            if self.comm is None:
                self._self_energy = engine.self_energy(np.array([0, total_iterations], dtype=np.int64))
            else:
                rank = self.comm.Get_rank()
                size = self.comm.Get_size()
                # Calculate the number of iterations per process
                iterations_per_process = total_iterations // size
                remainder = total_iterations % size
                # Calculate the start and end indices for this process
                start_index = rank * iterations_per_process + min(rank, remainder)
                end_index = start_index + iterations_per_process + (1 if rank < remainder else 0)
                local_energy = engine.self_energy(np.array([start_index, end_index], dtype=np.int64))
                global_energy = self.comm.reduce(local_energy, op=MPI.SUM, root=0)
                global_energy = self.comm.bcast(global_energy, root=0)
                self._self_energy = global_energy
        return self._self_energy

    def solve_induced_dipoles(self,
                              threshold: float = 1e-8,
                              max_iterations: int = 100,
                              solver: Optional[str] = 'jacobi',
                              external_fields: Optional[np.ndarray] = None,
                              perturbed: bool = False
                              ) -> None | np.ndarray:
        """Solves for the induced dipoles on all atoms.

        Args:
            threshold: Convergence threshold.
            max_iterations: Maximum number of iterations.
            solver: Type of solver used.
            external_fields: External fields that contribute additionally to the internal fields to induce dipoles.
            perturbed: Flag that indicates if induced dipoles from perturbed fields are to be calculated.
        """
        if external_fields is not None:
            if not perturbed:
                static_fields = self.multipole_fields + external_fields
            else:
                static_fields = external_fields
        else:
            static_fields = self.multipole_fields
            external_fields = np.zeros([self.num_atoms, 3])
        # First guess for induced dipoles
        if not perturbed:
            if np.all(self.induced_dipoles.induced_dipoles == 0):
                starting_guess = np.zeros([self.num_atoms, 3])
                for i, field in enumerate(static_fields):
                    starting_guess[i, :] = np.einsum('ij, j', self.polarizabilities[i], field)
            else:
                residue_norm = np.linalg.norm(external_fields - self.induced_dipoles.external_fields)
                max_residue_norm = np.max(np.abs(external_fields - self.induced_dipoles.external_fields))
                if residue_norm == 0 and max_residue_norm == 0:
                    print("Residue norm between new and old external fields is 0, induced dipoles will not be "
                          "recalculated.")
                    return
                elif residue_norm < 1e-6 and max_residue_norm < 1e-6:
                    print("Residue norm between new and old external fields is smaller than 1e-6, old induced dipoles will "
                          "be used as a starting guess.")
                    starting_guess = self.induced_dipoles.induced_dipoles
                else:
                    print("Residue norm between new and old external fields is larger than 1e-6, old induced dipoles will "
                          "not be used as a starting guess.")
                    starting_guess = np.zeros([self.num_atoms, 3])
                    for i, field in enumerate(static_fields):
                        starting_guess[i, :] = np.einsum('ij, j', self.polarizabilities[i], field)
        else:
            # check perturbed induced dipoles
            residue_norms = []
            for old_pert_induced_dipole in self.perturbed_induced_dipoles:
                residue_norm = np.linalg.norm(external_fields - old_pert_induced_dipole.external_fields)
                if residue_norm == 0:
                    print("Residue norm between new and old external fields is 0, induced dipoles will not be "
                          "recalculated.")
                    return old_pert_induced_dipole.induced_dipoles
                else:
                    residue_norms.append(residue_norm)
            # check which residue norm is the smallest
            min_res_norm = min(residue_norms)
            if min_res_norm < 1e-6 :
                print(
                    "Residue norm between new and old external fields is smaller than 1e-6, old induced dipoles will "
                    "be used as a starting guess.")
                starting_guess = self.perturbed_induced_dipoles[residue_norms.index(min_res_norm)].induced_dipoles
            else:
                print(
                    "Residue norm between new and old external fields is larger than 1e-6, old induced dipoles will "
                    "not be used as a starting guess.")
                starting_guess = np.zeros([self.num_atoms, 3])
                for i, field in enumerate(static_fields):
                    starting_guess[i, :] = np.einsum('ij, j', self.polarizabilities[i], field)
        induced_dipoles, num_iter = None, None
        if solver == 'jacobi':
            induced_dipoles, num_iter = solvers.induced_dipoles_jacobi(coordinates=self.coordinates,
                                                                       polarizabilities=self.polarizabilities,
                                                                       exclusions=self.exclusions,
                                                                       indices=self.indices,
                                                                       fields=static_fields,
                                                                       starting_guess=starting_guess,
                                                                       threshold=threshold,
                                                                       max_iterations=max_iterations,
                                                                       comm=self.comm)
        k = 0
        for fragment in self.classical_fragments:
            for atom in fragment.atoms:
                atom.induced_dipole = induced_dipoles[k]
                k += 1
        if not perturbed:
            self.induced_dipoles = InducedDipoles(induced_dipoles=induced_dipoles,
                                                  external_fields=external_fields,
                                                  number_of_iterations=num_iter,
                                                  solver=solver,
                                                  threshold=threshold)
        else:
            # FIXME Maybe this will cache too much if the property is of too high order?
            self.perturbed_induced_dipoles.append(InducedDipoles(induced_dipoles=induced_dipoles,
                                                                 external_fields=external_fields,
                                                                 number_of_iterations=num_iter,
                                                                 solver=solver,
                                                                 threshold=threshold))
            return induced_dipoles


@dataclass
class InducedDipoles(ClassicalSubsystem):
    """DataClass for induced dipoles. Contains array of induced dipoles, the external fields used to induce these
    dipoles, the field produced by the induced dipoles, the number of iterations it took to calculate the induced
    dipoles, and the name of the solver used.
    """
    induced_dipoles: np.ndarray
    external_fields: np.ndarray
    number_of_iterations: int
    solver: str
    threshold: float


class ContinuumSubsystem(Subsystem):
    """A ContinuumSubsystem represents a dielectric continuum.
    """
    pass
