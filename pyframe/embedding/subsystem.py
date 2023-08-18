from __future__ import annotations

import json
import numpy as np
from typing import Optional
from pyframe.embedding import fragment, particle, density_matrix
from pathlib import Path


class Subsystem:
    """A Subsystem represents a subsystem of the whole system by partitioning the system through the multiscale modeling
    approach.
    """

    def __init__(self,
                 name: Optional[str]):
        self._name = name


class QuantumSubsystem(Subsystem):
    """A QuantumSubsystem represents a collection of QuantumFragments, Nuclei and DensityMatrices.

    Args:
        input_data: Filepath to JSON file that contains the input data.
        name: Name of the QuantumSubsystem.
    """

    def __init__(self,
                 input_data: Path | str,
                 name: Optional[str] = None,
                 ):
        Subsystem.__init__(self, name=name)
        with open(input_data) as json_file:
            self._input_data = json.load(json_file).get('quantum_subsystem', None)
        if self._input_data.get('nuclei', None) is not None:
            self.nuclei = []
            for nuclei in self._input_data['nuclei']:
                nuclei['coordinate'] = np.array(nuclei['coordinate'])
                self.nuclei.append(particle.Nucleus(**nuclei))
        if self._input_data.get('quantum_fragments', None) is not None:
            self.quantum_fragments = []
            for frag in self._input_data['quantum_fragments']:
                self.quantum_fragments.append(fragment.QuantumFragment(**frag))
        if self._input_data.get('density_matrix', None) is not None:
            self.density_matrix = density_matrix.DensityMatrix(self._input_data['density_matrix'])
        else:
            self.density_matrix = density_matrix.DensityMatrix(np.zeros(1))

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
                 input_data: Path | str,
                 name: Optional[str] = None
                 ):
        Subsystem.__init__(self, name=name)
        with open(input_data) as json_file:
            self._input_data = json.load(json_file).get('classical_subsystem', None)
        if self._input_data.get('atoms', None) is not None:
            self.atoms = []
            for n in self._input_data['atoms']:
                n['coordinate'] = np.array(n['coordinate'])
                self.atoms.append(particle.Atom(**n))
        if self._input_data.get('classical_fragments', None) is not None:
            self.classical_fragments = []
            for f in self._input_data['classical_fragments']:
                self.classical_fragments.append(fragment.ClassicalFragment(**f))

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
        if hasattr(self, 'classical_fragments'):
            for fragments in self.classical_fragments:
                pot.append(fragments.potential(coordinate=coordinate,
                                               pot_derivative_order=pot_derivative_order,
                                               origin_derivative_order=origin_derivative_order,
                                               coord_multipole_order=coord_multipole_order))
        if hasattr(self, 'atoms'):
            for atoms in self.atoms:
                pot.append(atoms.potential(coordinate=coordinate,
                                           pot_derivative_order=pot_derivative_order,
                                           origin_derivative_order=origin_derivative_order,
                                           coord_multipole_order=coord_multipole_order))
        if array_of_potentials is False:
            return np.array(sum(pot))
        if array_of_potentials is True:
            return np.array(pot)


class ContinuumSubsystem(Subsystem):
    """A ContinuumSubsystem represents a dielectric continuum.
    """
    pass
