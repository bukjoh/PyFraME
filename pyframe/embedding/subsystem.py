from __future__ import annotations

import numpy as np
from mpi4py import MPI
from dataclasses import dataclass
from typing import Any

from pyframe.embedding import engine
from .polytensor import FirstDegreePolytensor
from .tensor_tools import uncompress_symmetric_matrix
from .solvers import induced_dipoles_jacobi
from .particle import Nucleus
from .fragment import QuantumFragment, ClassicalFragment


class Subsystem:
    """A Subsystem represents a subsystem of the whole system by partitioning the system through the multiscale modeling
    approach.
    """

    def __init__(self,
                 name: str | None = None,
                 comm: MPI.Comm | None = None
                 ) -> None:
        self.name = name
        self.comm = comm


class QuantumSubsystem(Subsystem):
    """A QuantumSubsystem represents a collection of QuantumFragments, Nuclei and DensityMatrices.

    Args:
        nuclei: List of Nuclei.
        quantum_fragments: Fragments of the QuantumSubsystem.
        name: Name of the QuantumSubsystem.
        comm: The MPI communicator.
    """

    def __init__(self,
                 nuclei: list[Nucleus] | None = None,
                 quantum_fragments: list[QuantumFragment] | None = None,
                 name: str | None = None,
                 comm: MPI.Comm | None = None
                 ) -> None:
        Subsystem.__init__(self, name=name, comm=comm)
        if nuclei is None:
            self.nuclei = []
        else:
            self.nuclei = nuclei
        self.num_nuclei = len(self.nuclei)
        self.quantum_fragments = quantum_fragments
        self._rep_lj_sigma = None
        self._rep_lj_epsilon = None
        self._disp_lj_sigma = None
        self._disp_lj_epsilon = None
        self._coordinates = None
        self._charges = None
        if self.comm is not None:
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()

    def __repr__(self):
        return f"<QuantumSubsystem {self.name}>"

    @property
    def coordinates(self) -> np.ndarray:
        if self._coordinates is None:
            self._coordinates = np.zeros([self.num_nuclei, 3], dtype=np.float64)
            for i, nucleus in enumerate(self.nuclei):
                self._coordinates[i, :] = nucleus.coordinate[:]
        return self._coordinates

    @property
    def charges(self) -> np.ndarray:
        if self._charges is None:
            self._charges = np.zeros(self.num_nuclei, dtype=np.float64)
            for i, nucleus in enumerate(self.nuclei):
                self.charges[i] = nucleus.charge[0]
        return self._charges

    @property
    def rep_lj_sigma(self) -> np.ndarray:
        if self._rep_lj_sigma is None:
            self._rep_lj_sigma = np.zeros([self.num_nuclei], dtype=np.float64)
            for i, nucleus in enumerate(self.nuclei):
                self._rep_lj_sigma[i] = nucleus.rep_parameters['lj_sigma']
        return self._rep_lj_sigma

    @property
    def rep_lj_epsilon(self) -> np.ndarray:
        if self._rep_lj_epsilon is None:
            self._rep_lj_epsilon = np.zeros([self.num_nuclei], dtype=np.float64)
            for i, nucleus in enumerate(self.nuclei):
                self._rep_lj_epsilon[i] = nucleus.rep_parameters['lj_epsilon']
        return self._rep_lj_epsilon

    @property
    def disp_lj_sigma(self) -> np.ndarray:
        if self._disp_lj_sigma is None:
            self._disp_lj_sigma = np.zeros([self.num_nuclei], dtype=np.float64)
            for i, nucleus in enumerate(self.nuclei):
                self._disp_lj_sigma[i] = nucleus.disp_parameters['lj_sigma']
        return self._disp_lj_sigma

    @property
    def disp_lj_epsilon(self) -> np.ndarray:
        if self._disp_lj_epsilon is None:
            self._disp_lj_epsilon = np.zeros([self.num_nuclei], dtype=np.float64)
            for i, nucleus in enumerate(self.nuclei):
                self._disp_lj_epsilon[i] = nucleus.disp_parameters['lj_epsilon']
        return self._disp_lj_epsilon

    def compute_nuclear_fields(self,
                               coordinates
                               ) -> np.ndarray:
        """Calculate the electric field from the nuclei at the given coordinates.

        Args:
            coordinates: Array of coordinates at which the field is calculated.

        Returns:
            Array of nuclear fields in the same ordering as the input coordinates.
        """
        engine.set_coords_nuc_coords_charges(coordinates, self.charges, self.coordinates)
        if self.comm is not None:
            avg, res = divmod(len(coordinates), self.size)
            counts = [avg + 1 if p < res else avg for p in range(self.size)]
            start = sum(counts[:self.rank])
            end = sum(counts[:self.rank + 1])
            nuclear_fields_global = np.zeros([len(coordinates), 3])
            nuclear_fields_local = engine.nuclei_fields(np.array([start, end], dtype=np.int64))
            self.comm.Allreduce([nuclear_fields_local, MPI.DOUBLE],
                                [nuclear_fields_global, MPI.DOUBLE],
                                op=MPI.SUM)
            return nuclear_fields_global
        else:
            return engine.nuclei_fields(np.array([0, len(coordinates)], dtype=np.int64))

    def compute_nuclear_field_gradients(self,
                                        coordinates
                                        ) -> np.ndarray:
        """Calculate the electric field gradient from the nuclei at the given coordinates.

        Args:
            coordinates: Array of coordinates at which the field gradients are calculated.

        Returns:
            Array of nuclear field gradients in the same ordering as the input coordinates.
                Shape: (number of nuclei, number of atoms, 6)
        """
        engine.set_coords_nuc_coords_charges(coordinates, self.charges, self.coordinates)
        if self.comm is not None:
            avg, res = divmod(len(coordinates), self.size)
            counts = [avg + 1 if p < res else avg for p in range(self.size)]
            start = sum(counts[:self.rank])
            end = sum(counts[:self.rank + 1])
            nuclear_fields_global = np.zeros([len(coordinates), 3])
            nuclear_fields_local = engine.nuclei_field_gradients(np.array([start, end], dtype=np.int64))
            self.comm.Allreduce([nuclear_fields_local, MPI.DOUBLE],
                                [nuclear_fields_global, MPI.DOUBLE],
                                op=MPI.SUM)
            return nuclear_fields_global
        else:
            return engine.nuclei_field_gradients(np.array([0, len(coordinates)], dtype=np.int64))

    def compute_electronic_fields(self,
                                  coordinates: np.ndarray,
                                  density_matrix: np.ndarray,
                                  integral_driver: Any
                                  ) -> np.ndarray:
        """Calculate the electric field from the electron density at the given coordinates.

        Args:
            coordinates: Coordinates on which the fields are to be evaluated.
            density_matrix: Density Matrix that is the source of the electronic field.
            integral_driver: Integral driver that calculates the electronic fields on coordinates.

        Returns:
            Electronic fields. Shape: (number of atoms, 3)
        """
        return integral_driver.electronic_fields(coordinates=coordinates, density_matrix=density_matrix)

    def compute_electronic_field_gradients(self,
                                           coordinates: np.ndarray,
                                           density_matrix: np.ndarray,
                                           integral_driver: Any
                                           ) -> np.ndarray:
        """Calculate the electric field gradients from the electron density at the given coordinates.

        Args:
            coordinates: Coordinates on which the fields are to be evaluated.
            density_matrix: Density Matrix that is the source of the electronic field.
            integral_driver: Integral driver that calculates the electronic fields on coordinates.

        Returns:
            Electronic field gradients.
                Shape: (number of nuclei, number of atoms, 6)
        """
        return integral_driver.electronic_field_gradient(coordinates=coordinates,
                                                         density_matrix=density_matrix)


class ClassicalSubsystem(Subsystem):
    """A ClassicalSubsystem represents a collection of ClassicalFragments and/or Particles.

    Args:
        classical_fragments: Fragments of the ClassicalSubsystem.
        name: Name of the ClassicalSubsystem.
        comm: The MPI communicator.
    """

    def __init__(self,
                 classical_fragments: list[ClassicalFragment],
                 name: str | None = None,
                 comm: MPI.Comm | None = None
                 ) -> None:
        if not classical_fragments:
            raise ValueError("ClassicalSubsystem must have at least one ClassicalFragment.")
        Subsystem.__init__(self, name=name, comm=comm)
        self.classical_fragments = classical_fragments
        self.num_atoms = 0
        for fragments in self.classical_fragments:
            self.num_atoms += len(fragments.atoms)
        self._coordinates = None
        self._multipole_orders = None
        self._degenerate_multipoles_with_taylor_coefficients = None
        self._indices = None
        self._exclusions = None
        self._charges = None
        self._dipole_dipole_polarizabilities = None
        self._rep_lj_sigma = None
        self._rep_lj_epsilon = None
        self._disp_lj_sigma = None
        self._disp_lj_epsilon = None
        self._multipole_fields = None
        self._environment_energy = None
        self.induced_dipoles = InducedDipoles(induced_dipoles=np.zeros([self.num_atoms, 3]),
                                              external_fields=np.zeros([self.num_atoms, 3]),
                                              number_of_iterations=0,
                                              threshold=1e-8,
                                              solver="None")
        self.perturbed_induced_dipoles = []
        if self.comm is not None:
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()

    def __repr__(self):
        return f"<ClassicalSubsystem {self.name}>"

    @property
    def coordinates(self):
        if self._coordinates is None:
            self._coordinates = np.zeros([self.num_atoms, 3], dtype=np.float64)
            k = 0
            for fragments in self.classical_fragments:
                for atom in fragments.atoms:
                    self._coordinates[k, :] = atom.coordinate[:]
                    k += 1
        return self._coordinates

    @property
    def multipole_orders(self) -> np.ndarray:
        if self._multipole_orders is None:
            self._multipole_orders = np.zeros([self.num_atoms], dtype=np.int64)
            k = 0
            for fragments in self.classical_fragments:
                for atom in fragments.atoms:
                    self._multipole_orders[k] = atom.multipole_order
                    k += 1
        return self._multipole_orders

    @property
    def degenerate_multipoles_with_taylor_coefficients(self) -> list:
        if self._degenerate_multipoles_with_taylor_coefficients is None:
            self._degenerate_multipoles_with_taylor_coefficients = []
            k = 0
            for fragments in self.classical_fragments:
                for atom in fragments.atoms:
                    self._degenerate_multipoles_with_taylor_coefficients.append(
                        FirstDegreePolytensor.multiply_elementwise(
                            atom.multipoles_with_degeneracy,
                            atom.taylor_coefficients).data)
                    k += 1
        return self._degenerate_multipoles_with_taylor_coefficients

    @property
    def indices(self) -> np.ndarray:
        if self._indices is None:
            self._indices = np.zeros(self.num_atoms, dtype=np.int64)
            k = 0
            for fragments in self.classical_fragments:
                for atom in fragments.atoms:
                    self._indices[k] = atom.index
                    k += 1
        return self._indices

    @property
    def exclusions(self):
        if self._exclusions is None:
            self._exclusions = []
            for fragments in self.classical_fragments:
                for atom in fragments.atoms:
                    self._exclusions.append(atom.exclusions)
        return self._exclusions

    @property
    def dipole_dipole_polarizabilities(self) -> np.ndarray:
        if self._dipole_dipole_polarizabilities is None:
            self._dipole_dipole_polarizabilities = np.zeros([self.num_atoms, 3, 3], dtype=np.float64)
            k = 0
            for fragments in self.classical_fragments:
                for atom in fragments.atoms:
                    # FIXME or set polarizability always to zeros when its not used? -> in the WRITE JSON is needs to
                    #  be 0's until the max polarizability order.
                    self._dipole_dipole_polarizabilities[k, :, :] = uncompress_symmetric_matrix(atom.
                                                                                                polarizability[4:10])
                    k += 1
        return self._dipole_dipole_polarizabilities

    @property
    def charges(self) -> np.ndarray:
        if self._charges is None:
            self._charges = np.zeros([self.num_atoms], dtype=np.float64)
            k = 0
            for fragments in self.classical_fragments:
                for atom in fragments.atoms:
                    self._charges[k] = atom.multipoles.data[0]
                    k += 1
        return self._charges

    @property
    def rep_lj_sigma(self) -> np.ndarray:
        if self._rep_lj_sigma is None:
            self._rep_lj_sigma = np.zeros([self.num_atoms], dtype=np.float64)
            k = 0
            for fragments in self.classical_fragments:
                for atom in fragments.atoms:
                    self._rep_lj_sigma[k] = atom.rep_parameters['lj_sigma']
                    k += 1
        return self._rep_lj_sigma

    @property
    def rep_lj_epsilon(self) -> np.ndarray:
        if self._rep_lj_epsilon is None:
            self._rep_lj_epsilon = np.zeros([self.num_atoms], dtype=np.float64)
            k = 0
            for fragments in self.classical_fragments:
                for atom in fragments.atoms:
                    self._rep_lj_epsilon[k] = atom.rep_parameters['lj_epsilon']
                    k += 1
        return self._rep_lj_epsilon

    @property
    def disp_lj_sigma(self) -> np.ndarray:
        if self._disp_lj_sigma is None:
            self._disp_lj_sigma = np.zeros([self.num_atoms], dtype=np.float64)
            k = 0
            for fragments in self.classical_fragments:
                for atom in fragments.atoms:
                    self._disp_lj_sigma[k] = atom.disp_parameters['lj_sigma']
                    k += 1
        return self._disp_lj_sigma

    @property
    def disp_lj_epsilon(self) -> np.ndarray:
        if self._disp_lj_epsilon is None:
            self._disp_lj_epsilon = np.zeros([self.num_atoms], dtype=np.float64)
            k = 0
            for fragments in self.classical_fragments:
                for atom in fragments.atoms:
                    self._disp_lj_epsilon[k] = atom.disp_parameters['lj_epsilon']
                    k += 1
        return self._disp_lj_epsilon

    @property
    def multipole_fields(self):
        if self._multipole_fields is None:
            self.compute_multipole_fields()
        return self._multipole_fields

    def compute_multipole_fields(self) -> None:
        """Compute the electric fields from all multipoles on all polarizable atoms."""
        engine.set_multipoles_multipoles_order(self.degenerate_multipoles_with_taylor_coefficients,
                                               self.multipole_orders)
        engine.set_coords_idcs_exlcs(self.coordinates,
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
            self.comm.Allreduce([multipole_fields_local, MPI.DOUBLE],
                                [self._multipole_fields, MPI.DOUBLE],
                                op=MPI.SUM)
        else:
            for i in range(len(self.coordinates)):
                self._multipole_fields[i, :] = engine.multipole_fields(np.array([i], dtype=np.int64)).T

    @property
    def environment_energy(self
                           ) -> float:
        """Calculate the internal electrostatic interaction energy between all multipoles.

        Returns:
            Self energy of the ClassicalSubsystem.
        """
        if self._environment_energy is None:
            engine.set_multipoles_multipoles_order(self.degenerate_multipoles_with_taylor_coefficients,
                                                   self.multipole_orders)
            engine.set_coords_idcs_exlcs(self.coordinates,
                                         self.indices,
                                         self.exclusions)
            total_iterations = (self.num_atoms - 1) * self.num_atoms // 2
            if self.comm is None:
                self._environment_energy = engine.environment_energy(np.array([0, total_iterations], dtype=np.int64))
            else:
                rank = self.comm.Get_rank()
                size = self.comm.Get_size()
                # Calculate the number of iterations per process
                iterations_per_process = total_iterations // size
                remainder = total_iterations % size
                # Calculate the start and end indices for this process
                start_index = rank * iterations_per_process + min(rank, remainder)
                end_index = start_index + iterations_per_process + (1 if rank < remainder else 0)
                local_energy = engine.environment_energy(np.array([start_index, end_index], dtype=np.int64))
                global_energy = self.comm.reduce(local_energy, op=MPI.SUM, root=0)
                global_energy = self.comm.bcast(global_energy, root=0)
                self._environment_energy = global_energy
        return self._environment_energy

    def solve_induced_dipoles(self,
                              threshold: float = 1e-8,
                              max_iterations: int = 100,
                              perturbed: bool = False,
                              solver: str = 'jacobi',
                              external_fields: np.ndarray | None = None
                              ) -> np.ndarray | None:
        """Solve for induced dipoles.

        Args:
            threshold: Convergence threshold.
            max_iterations: Maximum number of iterations.
            solver: Type of solver used.
            external_fields: External fields that are added to the internal permanent and induced fields.
            perturbed: Flag that indicates if induced dipoles from perturbed fields are to be calculated.
        """
        if external_fields is None:
            static_fields = np.zeros([self.num_atoms, 3])
        else:
            static_fields = external_fields
        if not perturbed:
            static_fields += self.multipole_fields
        # First guess for induced dipoles
        if not perturbed:
            if np.all(self.induced_dipoles.induced_dipoles == 0):
                starting_guess = np.zeros([self.num_atoms, 3])
                for i, field in enumerate(static_fields):
                    starting_guess[i, :] = np.einsum('ij, j', self.dipole_dipole_polarizabilities[i], field)
            else:
                residue_norm = np.linalg.norm(external_fields - self.induced_dipoles.external_fields)
                max_residue_norm = np.max(np.abs(external_fields - self.induced_dipoles.external_fields))
                if residue_norm == 0 and max_residue_norm == 0:
                    print("Residue norm between new and old external fields is 0, induced dipoles will not be "
                          "recalculated.")
                    return
                elif residue_norm < 1e-6 and max_residue_norm < 1e-6:
                    print("Residue norm between new and old external fields is smaller than 1e-6, old induced dipoles "
                          "will be used as a starting guess.")
                    starting_guess = self.induced_dipoles.induced_dipoles
                else:
                    print("Residue norm between new and old external fields is larger than 1e-6, old induced dipoles "
                          "will not be used as a starting guess.")
                    starting_guess = np.zeros([self.num_atoms, 3])
                    for i, field in enumerate(static_fields):
                        starting_guess[i, :] = np.einsum('ij, j', self.dipole_dipole_polarizabilities[i], field)
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
            if min_res_norm < 1e-6:
                print("Residue norm between new and old external fields is smaller than 1e-6, old induced dipoles will "
                      "be used as a starting guess.")
                starting_guess = self.perturbed_induced_dipoles[residue_norms.index(min_res_norm)].induced_dipoles
            else:
                print("Residue norm between new and old external fields is larger than 1e-6, old induced dipoles will "
                      "not be used as a starting guess.")
                starting_guess = np.zeros([self.num_atoms, 3])
                for i, field in enumerate(static_fields):
                    starting_guess[i, :] = np.einsum('ij, j', self.dipole_dipole_polarizabilities[i], field)
        induced_dipoles, num_iter = None, None
        if solver == 'jacobi':
            induced_dipoles, num_iter = induced_dipoles_jacobi(coordinates=self.coordinates,
                                                               polarizabilities=self.dipole_dipole_polarizabilities,
                                                               exclusions=self.exclusions,
                                                               indices=self.indices,
                                                               fields=static_fields,
                                                               starting_guess=starting_guess,
                                                               threshold=threshold,
                                                               max_iterations=max_iterations,
                                                               comm=self.comm)
        # TODO maybe remove this part?
        k = 0
        for fragment in self.classical_fragments:
            for atom in fragment.atoms:
                atom.induced_dipole = induced_dipoles[k]
                k += 1
        # TODO Maybe have only one list of induced dipoles.
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
