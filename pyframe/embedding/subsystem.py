from __future__ import annotations

import numpy as np
from mpi4py import MPI
from dataclasses import dataclass
from typing import Any

from pyframe.embedding import engine
from .polytensor import FirstDegreePolytensor
from .tensor_tools import uncompress_symmetric_matrix
from .solvers import (induced_dipoles_jacobi, induced_dipoles_jidiis, induced_dipoles_dcji, induced_dipoles_dcjidiis,
                      induced_dipoles_fmm)
from .particle import Nucleus
from .fragment import QuantumFragment, ClassicalFragment
from .logging_util import log_manager


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

    @coordinates.setter
    def coordinates(self, new_coordinates: np.ndarray):
        if not isinstance(new_coordinates, np.ndarray):
            raise TypeError("Coordinates must be a NumPy array.")
        if new_coordinates.shape != (self.num_nuclei, 3):
            raise ValueError(f"Coordinates must have shape ({self.num_nuclei}, 3).")

        self._coordinates = new_coordinates

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
            nuclear_fields = engine.nuclei_fields(np.array([start, end], dtype=np.int64))
            nuclear_fields = self.comm.allreduce(nuclear_fields)
            return nuclear_fields
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
            nuclear_field_gradients = engine.nuclei_field_gradients(np.array([start, end], dtype=np.int64))
            nuclear_field_gradients = self.comm.allreduce(nuclear_field_gradients)
            return nuclear_field_gradients
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
        if self.comm is not None:
            avg, res = divmod(len(coordinates), self.size)
            counts = [avg + 1 if p < res else avg for p in range(self.size)]
            start = sum(counts[:self.rank])
            end = sum(counts[:self.rank + 1])
            electronic_fields = np.zeros([len(coordinates), 3])
            electronic_fields[start:end] += -1.0 * integral_driver.electronic_fields(coordinates=coordinates[start:end],
                                                                                     density_matrix=density_matrix)
            electronic_fields = self.comm.allreduce(electronic_fields)
            return electronic_fields
        else:
            return -1.0 * integral_driver.electronic_fields(coordinates=coordinates, density_matrix=density_matrix)

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
        if self.comm is not None:
            avg, res = divmod(len(coordinates), self.size)
            counts = [avg + 1 if p < res else avg for p in range(self.size)]
            start = sum(counts[:self.rank])
            end = sum(counts[:self.rank + 1])
            electronic_field_gradient = np.zeros([len(coordinates), 3])
            electronic_field_gradient[start:end] += -1.0 * integral_driver.electronic_field_gradients(
                coordinates=coordinates[start:end],
                density_matrix=density_matrix)
            electronic_field_gradient = self.comm.allreduce(electronic_field_gradient)
            return electronic_field_gradient
        else:
            return -1.0 * integral_driver.electronic_field_gradients(coordinates=coordinates,
                                                                     density_matrix=density_matrix)


    def compute_electronic_induction_energy_gradient(self):
        return NotImplementedError


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
                    #  be 0's until the max polarizability order. -> polarizability should actually be a second degree polytensor created in atom class
                    # TODO change in atom class polarizability format to a second degree polytensor.
                    self._dipole_dipole_polarizabilities[k, :, :] = uncompress_symmetric_matrix(atom.
                                                                                                polarizability[4:10])
                    k += 1
        return self._dipole_dipole_polarizabilities

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
    def multipole_fields(self) -> np.ndarray:
        if self._multipole_fields is None:
            self.compute_multipole_fields()
        return self._multipole_fields

    def multipole_fields_fmm(self,
                             solver: str | None = None,
                             tree_ncrit: int = 64,
                             tree_expansion_order: int = 5,
                             theta: float = 0.5
                             ) -> np.ndarray:
        if self._multipole_fields is None:
            self.compute_multipole_fields(solver=solver,
                                          tree_ncrit=tree_ncrit,
                                          tree_expansion_order=tree_expansion_order,
                                          theta=theta)
        return self._multipole_fields

    def compute_multipole_fields(self,
                                 solver: str | None = None,
                                 tree_ncrit: int = 64,
                                 tree_expansion_order: int = 5,
                                 theta: float = 0.5
                                 ) -> None:
        """Compute the electric fields from all multipoles on all polarizable atoms."""
        engine.set_multipoles_multipole_orders(self.degenerate_multipoles_with_taylor_coefficients,
                                               self.multipole_orders)

        self._multipole_fields = np.zeros([self.num_atoms, 3])
        if solver == "fmm":
            shifted_exclusions = [
                tuple(value - 1 for value in exclusion) for exclusion in self.exclusions
            ]
            engine.set_coords_idcs_exlcs(self.coordinates,
                                         self.indices,
                                         shifted_exclusions)
            damping = 0.0
            self._multipole_fields -= engine.multipole_fields_fmm(tree_ncrit, tree_expansion_order, theta, damping)

        else:
            engine.set_coords_idcs_exlcs(self.coordinates,
                                         self.indices,
                                         self.exclusions)
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

    def environment_energy(self,
                           vdw_method: str = 'LJ',
                           vdw_combination_rule: str = 'Lorentz-Berthelot'
                           ) -> float:
        """Calculate the internal electrostatic, nonelectrostatic repulsion, and dispersion interaction energy between
        all multipoles.

        Returns:
            Environment energy.
        """
        if self._environment_energy is None:
            self._environment_energy = self.compute_electrostatic_energy()
            if vdw_method == 'LJ':
                self._environment_energy += self.compute_repulsion_energy(vdw_combination_rule)
                self._environment_energy += self.compute_dispersion_energy(vdw_combination_rule)
            else:
                raise NotImplementedError("This method has not been implemented yet.")
        return self._environment_energy

    def compute_electrostatic_energy(self) -> float:
        """Compute the electrostatic energy."""
        engine.set_multipoles_multipole_orders(self.degenerate_multipoles_with_taylor_coefficients,
                                               self.multipole_orders)
        engine.set_coords_idcs_exlcs(self.coordinates, self.indices, self.exclusions)
        total_iterations = (self.num_atoms - 1) * self.num_atoms // 2
        return self._compute_environment_energy_distributed_or_serial(
            lambda start, end: engine.electrostatic_environment_energy(np.array([start, end], dtype=np.int64)),
            total_iterations
        )

    def compute_repulsion_energy(self, vdw_combination_rule) -> float:
        """Compute the repulsion energy."""
        if vdw_combination_rule == 'Lorentz-Berthelot':
            engine.set_combination_rule(vdw_combination_rule)
        else:
            raise NotImplementedError("This combination rule has not been implemented yet.")
        engine.set_lj_classical_sigma_epsilon(self.rep_lj_sigma, self.rep_lj_epsilon)
        engine.set_coords_idcs_exlcs(self.coordinates, self.indices, self.exclusions)
        total_iterations = (self.num_atoms - 1) * self.num_atoms // 2
        return self._compute_environment_energy_distributed_or_serial(
            lambda start, end: engine.lj_repulsion_environment_energy(np.array([start, end], dtype=np.int64)),
            total_iterations
        )

    def compute_dispersion_energy(self, vdw_combination_rule) -> float:
        """Compute the dispersion energy."""
        if vdw_combination_rule == 'Lorentz-Berthelot':
            engine.set_combination_rule(vdw_combination_rule)
        else:
            raise NotImplementedError("This combination rule has not been implemented yet.")
        engine.set_lj_classical_sigma_epsilon(self.disp_lj_sigma, self.disp_lj_epsilon)
        engine.set_coords_idcs_exlcs(self.coordinates, self.indices, self.exclusions)
        total_iterations = (self.num_atoms - 1) * self.num_atoms // 2
        return self._compute_environment_energy_distributed_or_serial(
            lambda start, end: engine.lj_dispersion_environment_energy(np.array([start, end], dtype=np.int64)),
            total_iterations
        )

    def _compute_environment_energy_distributed_or_serial(self, compute_fn, total_iterations: int) -> float:
        """
        Compute energy either in serial or in parallel using MPI.

        Args:
            compute_fn (function): A function that computes energy for a given range.
            total_iterations (int): Total number of iterations.

        Returns:
            float: Computed energy.
        """
        if self.comm is None:
            # Serial computation
            log_manager.logger.debug(
                print("Calculation of environment energy in serial.")
            )
            return compute_fn(0, total_iterations)
        else:
            # Parallel computation
            rank = self.comm.Get_rank()
            size = self.comm.Get_size()

            iterations_per_process = total_iterations // size
            remainder = total_iterations % size

            # Calculate the start and end indices for this process
            start_index = rank * iterations_per_process + min(rank, remainder)
            end_index = start_index + iterations_per_process + (1 if rank < remainder else 0)

            # Compute local energy
            local_energy = compute_fn(start_index, end_index)

            # Reduce energy across all processes
            global_energy = self.comm.reduce(local_energy, op=MPI.SUM, root=0)
            return self.comm.bcast(global_energy, root=0)

    def solve_induced_dipoles(self,
                              threshold: float = 1e-8,
                              max_iterations: int = 100,
                              mic: bool = False,
                              box: np.ndarray = np.array([]),
                              solver: str = 'jidiis',
                              max_diis: int = 5,
                              init_diis: int = 3,
                              k_cluster: int = 5,
                              cluster_size_range: int = -1,
                              exclude_static_internal_fields: bool = False,
                              external_fields: np.ndarray | None = None,
                              tree_ncrit: int = 64,
                              tree_expansion_order: int = 5,
                              theta: float = 0.5
                              ) -> None:
        """Solve for induced dipoles.

        Args:
            threshold: Convergence threshold.
            max_iterations: Maximum number of iterations.
            mic: Boolean indicating whether induced dipoles are calculated using mic scaled coordinates.
            box: Dimensions of the simulation box described by vectors.
            solver: Type of solver used.
            max_diis: Maximum number of previous iterations to consider in the DIIS method.
            init_diis: Iteration number at which DIIS is initiated.
            k_cluster: The number of clusters to form in K-means clustering, used for divide and conquer methods.
            cluster_size_range: The range of cluster size deviations from the mean cluster size in number of atoms. -1
            allows cluster deviations of any size.
            exclude_static_internal_fields: Exclude any static internal fields, e.g., from permanent multipoles.
            external_fields: External fields that are added to the internal permanent and induced fields.
            tree_ncrit: FMM parameter: Maximum number of particles per tree node.
            tree_expansion_order: FMM parameter: Expansion order for tree-based summation schemes.
            theta: FMM parameter: Opening angle for tree-based summation schemes.
        """
        if external_fields is None:
            static_fields = np.zeros([self.num_atoms, 3])
        else:
            static_fields = external_fields
        if not exclude_static_internal_fields:
            if solver == 'fmm':
                static_fields += self.multipole_fields_fmm(solver=solver,
                                                           tree_ncrit=tree_ncrit,
                                                           tree_expansion_order=tree_expansion_order,
                                                           theta=theta)
            else:
                static_fields += self.multipole_fields
        # First guess for induced dipoles
        if np.all(self.induced_dipoles.induced_dipoles == 0):
            starting_guess = np.zeros([self.num_atoms, 3])
            for i, field in enumerate(static_fields):
                starting_guess[i, :] = np.einsum('ij, j', self.dipole_dipole_polarizabilities[i], field)
        else:
            residue_norm = np.linalg.norm(external_fields - self.induced_dipoles.external_fields)
            max_residue_norm = np.max(np.abs(external_fields - self.induced_dipoles.external_fields))
            if residue_norm == 0 and max_residue_norm == 0:
                log_manager.logger.debug(
                    "Residue norm between new and old external fields is 0, induced dipoles will not be "
                    "recalculated.")
                return
            elif residue_norm < 1e-6 and max_residue_norm < 1e-6:
                log_manager.logger.debug(
                    "Residue norm between new and old external fields is smaller than 1e-6, old induced dipoles "
                    "will be used as a starting guess.")
                starting_guess = self.induced_dipoles.induced_dipoles
            else:
                log_manager.logger.debug(
                    "Residue norm between new and old external fields is larger than 1e-6, old induced dipoles "
                    "will not be used as a starting guess.")
                starting_guess = np.zeros([self.num_atoms, 3])
                for i, field in enumerate(static_fields):
                    starting_guess[i, :] = np.einsum('ij, j', self.dipole_dipole_polarizabilities[i], field)
        if mic and np.shape(box) != (3, 3):
            log_manager.logger.debug(
                print("No simulation box dimensions, mic disabled.")
            )
            mic = False
        induced_dipoles, num_iter = None, None
        if solver == 'jacobi':
            induced_dipoles, num_iter = induced_dipoles_jacobi(coordinates=self.coordinates,
                                                               polarizabilities=self.dipole_dipole_polarizabilities,
                                                               exclusions=self.exclusions,
                                                               indices=self.indices,
                                                               fields=static_fields,
                                                               starting_guess=starting_guess,
                                                               mic=mic,
                                                               box=box,
                                                               threshold=threshold,
                                                               max_iterations=max_iterations,
                                                               comm=self.comm)
        elif solver == 'jidiis':
            induced_dipoles, num_iter = induced_dipoles_jidiis(coordinates=self.coordinates,
                                                               polarizabilities=self.dipole_dipole_polarizabilities,
                                                               exclusions=self.exclusions,
                                                               indices=self.indices,
                                                               fields=static_fields,
                                                               starting_guess=starting_guess,
                                                               mic=mic,
                                                               box=box,
                                                               threshold=threshold,
                                                               max_iterations=max_iterations,
                                                               max_diis=max_diis,
                                                               init_diis=init_diis,
                                                               comm=self.comm)
        elif solver == 'dcji':
            induced_dipoles, num_iter = induced_dipoles_dcji(coordinates=self.coordinates,
                                                             polarizabilities=self.dipole_dipole_polarizabilities,
                                                             exclusions=self.exclusions,
                                                             indices=self.indices,
                                                             fields=static_fields,
                                                             starting_guess=starting_guess,
                                                             mic=mic,
                                                             box=box,
                                                             threshold=threshold,
                                                             max_iterations=max_iterations,
                                                             k_cluster=k_cluster,
                                                             cluster_size_range=cluster_size_range,
                                                             comm=self.comm)
        elif solver == 'dcjidiis':
            induced_dipoles, num_iter = induced_dipoles_dcjidiis(coordinates=self.coordinates,
                                                                 polarizabilities=self.dipole_dipole_polarizabilities,
                                                                 exclusions=self.exclusions,
                                                                 indices=self.indices,
                                                                 fields=static_fields,
                                                                 starting_guess=starting_guess,
                                                                 mic=mic,
                                                                 box=box,
                                                                 threshold=threshold,
                                                                 max_iterations=max_iterations,
                                                                 max_diis=max_diis,
                                                                 init_diis=init_diis,
                                                                 k_cluster=k_cluster,
                                                                 cluster_size_range=cluster_size_range,
                                                                 comm=self.comm)
        elif solver == 'fmm':
            induced_dipoles, num_iter = induced_dipoles_fmm(coordinates=self.coordinates,
                                                            polarizabilities=self.dipole_dipole_polarizabilities,
                                                            exclusions=self.exclusions,
                                                            indices=self.indices,
                                                            fields=static_fields,
                                                            starting_guess=starting_guess,
                                                            mic=mic,
                                                            box=box,
                                                            threshold=threshold,
                                                            max_iterations=max_iterations,
                                                            tree_ncrit=tree_ncrit,
                                                            tree_expansion_order=tree_expansion_order,
                                                            theta=theta,
                                                            comm=self.comm)
        # TODO maybe remove this part? -> MPI parallelize it?
        k = 0
        for fragment in self.classical_fragments:
            for atom in fragment.atoms:
                atom.induced_dipole = induced_dipoles[k]
                k += 1
        self.induced_dipoles = InducedDipoles(induced_dipoles=induced_dipoles,
                                              external_fields=external_fields,
                                              number_of_iterations=num_iter,
                                              solver=solver,
                                              threshold=threshold)

    def solve_perturbed_induced_dipoles(self,
                                        threshold: float = 1e-8,
                                        max_iterations: int = 100,
                                        mic: bool = False,
                                        box: np.ndarray = np.array([]),
                                        solver: str = 'jacobi',
                                        max_diis: int = 5,
                                        init_diis: int = 3,
                                        k_cluster: int = 5,
                                        cluster_size_range: int = -1,
                                        external_fields: np.ndarray | None = None
                                        ) -> np.ndarray:
        """Solve for perturbed induced dipoles.

        Args:
            threshold: Convergence threshold.
            max_iterations: Maximum number of iterations.
            mic: Boolean indicating whether induced dipoles are calculated using mic scaled coordinates.
            box: Dimensions of the simulation box described by vectors.
            solver: Type of solver used.
            max_diis: Maximum number of previous iterations to consider in the DIIS method.
            init_diis: Iteration number at which DIIS is initiated.
            k_cluster: The number of clusters to form in K-means clustering, used for divide and conquer methods.
            cluster_size_range: The range of cluster size deviations from the mean cluster size in number of atoms. -1
            allows cluster deviations of any size.
            external_fields: External fields that are added to the internal permanent and induced fields.
        """
        if external_fields is None:
            static_fields = np.zeros([self.num_atoms, 3])
        else:
            static_fields = external_fields
        # First guess for induced dipoles
        # check perturbed induced dipoles
        residue_norms = []
        for old_pert_induced_dipole in self.perturbed_induced_dipoles:
            residue_norm = np.linalg.norm(external_fields - old_pert_induced_dipole.external_fields)
            if residue_norm == 0:
                log_manager.logger.debug(
                    print("Residue norm between new and old external fields is 0, induced dipoles will not be "
                          "recalculated.")
                )
                return old_pert_induced_dipole.induced_dipoles
            else:
                residue_norms.append(residue_norm)
        # check which residue norm is the smallest
        min_res_norm = min(residue_norms)
        if min_res_norm < 1e-6:
            log_manager.logger.debug(
                print("Residue norm between new and old external fields is smaller than 1e-6, old induced dipoles will "
                      "be used as a starting guess.")
            )
            starting_guess = self.perturbed_induced_dipoles[residue_norms.index(min_res_norm)].induced_dipoles
        else:
            log_manager.logger.debug(
                print("Residue norm between new and old external fields is larger than 1e-6, old induced dipoles will "
                      "not be used as a starting guess.")
            )
            starting_guess = np.zeros([self.num_atoms, 3])
            for i, field in enumerate(static_fields):
                starting_guess[i, :] = np.einsum('ij, j', self.dipole_dipole_polarizabilities[i], field)
        if mic and np.shape(box) != (3, 3):
            log_manager.logger.debug(
                print("No simulation box dimensions, mic disabled.")
            )
            mic = False
        induced_dipoles, num_iter = None, None
        if solver == 'jacobi':
            induced_dipoles, num_iter = induced_dipoles_jacobi(coordinates=self.coordinates,
                                                               polarizabilities=self.dipole_dipole_polarizabilities,
                                                               exclusions=self.exclusions,
                                                               indices=self.indices,
                                                               fields=static_fields,
                                                               starting_guess=starting_guess,
                                                               mic=mic,
                                                               box=box,
                                                               threshold=threshold,
                                                               max_iterations=max_iterations,
                                                               comm=self.comm)
        elif solver == 'jidiis':
            induced_dipoles, num_iter = induced_dipoles_jidiis(coordinates=self.coordinates,
                                                               polarizabilities=self.dipole_dipole_polarizabilities,
                                                               exclusions=self.exclusions,
                                                               indices=self.indices,
                                                               fields=static_fields,
                                                               starting_guess=starting_guess,
                                                               mic=mic,
                                                               box=box,
                                                               threshold=threshold,
                                                               max_iterations=max_iterations,
                                                               max_diis=max_diis,
                                                               init_diis=init_diis,
                                                               comm=self.comm)
        elif solver == 'dcji':
            induced_dipoles, num_iter = induced_dipoles_dcji(coordinates=self.coordinates,
                                                             polarizabilities=self.dipole_dipole_polarizabilities,
                                                             exclusions=self.exclusions,
                                                             indices=self.indices,
                                                             fields=static_fields,
                                                             starting_guess=starting_guess,
                                                             mic=mic,
                                                             box=box,
                                                             threshold=threshold,
                                                             max_iterations=max_iterations,
                                                             k_cluster=k_cluster,
                                                             cluster_size_range=cluster_size_range,
                                                             comm=self.comm)
        elif solver == 'dcjidiis':
            induced_dipoles, num_iter = induced_dipoles_dcjidiis(coordinates=self.coordinates,
                                                                 polarizabilities=self.dipole_dipole_polarizabilities,
                                                                 exclusions=self.exclusions,
                                                                 indices=self.indices,
                                                                 fields=static_fields,
                                                                 starting_guess=starting_guess,
                                                                 mic=mic,
                                                                 box=box,
                                                                 threshold=threshold,
                                                                 max_iterations=max_iterations,
                                                                 max_diis=max_diis,
                                                                 init_diis=init_diis,
                                                                 k_cluster=k_cluster,
                                                                 cluster_size_range=cluster_size_range,
                                                                 comm=self.comm)

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
