from __future__ import annotations

import numpy as np
from typing import Optional
from pyframe.embedding import particle, density_matrix


class Fragment:
    """A Fragment object represents a collection of particle objects of a Classical- or Quantum-subsystem, which are
    Classical fragments and Quantum fragments, respectively.
    Args:
        index: Index of the fragment.
        name: Name of the fragment.
    """

    def __init__(self,
                 index: int,
                 name: Optional[str] = None
                 ):
        self._index = index
        self._name = name


class ClassicalFragment(Fragment):
    """A ClassicalFragment object represents a collection of particle objects of a Classical subsystem.
    Args:
        index: Index of the fragment.
        atoms: List of dictionaries of all the particles in the fragment. Each dictionary must contain the type
        of particle and all the **kwargs, respectively.
        name: Name of the fragment.
    """

    def __init__(self,
                 index: int,
                 atoms: list,
                 name: Optional[str] = None,
                 ):
        Fragment.__init__(self, index=index, name=name)
        self.atoms = []
        for a in atoms:
            a['coordinate'] = np.array(a['coordinate'])
            self.atoms.append(particle.Atom(**a))
        self.num_atoms = len(atoms)

    def potential(self,
                  coordinate: np.ndarray,
                  pot_derivative_order: Optional[int] = 0,
                  origin_derivative_order: Optional[int] = 0,
                  coord_multipole_order: Optional[int] = 0
                  ) -> float | np.ndarray:
        """Calculates the sum of electrostatic potential and its derivatives of the atoms in a ClassicalFragment.

        Args:
            coordinate: Coordinates at which the potential is evaluated.
            pot_derivative_order: Order of the derivative of the potential.
            origin_derivative_order: Order of derivative with respect to the origin of the potential.
            coord_multipole_order: Multipole order at coordinate.
        Returns:
            Electrostatic potential or its derivative of the fragment at coordinates. If coord_multipole_order is given,
            the derivatives with respect to the charge or multipole at coordinate are included.
        """
        pot = []
        for atoms in self.atoms:
            pot.append(atoms.potential(coordinate=coordinate,
                                       pot_derivative_order=pot_derivative_order,
                                       origin_derivative_order=origin_derivative_order,
                                       coord_multipole_order=coord_multipole_order))
        return np.array(sum(pot))


class QuantumFragment(Fragment):
    """A QuantumFragment object represents a collection of particle objects and electron density of a Quantum subsystem.
    Args:
        index: Index of the fragment.
        nuclei: List of dictionaries of all the particles in the fragment. Each dictionary must contain the type
        of particle and all the **kwargs, respectively.
        e_density_matrix: Electron density of the fragment.
        name: Name of the fragment.
    """

    def __init__(self,
                 index: int,
                 nuclei: list,
                 e_density_matrix: density_matrix.DensityMatrix,
                 name: Optional[str] = None
                 ):
        Fragment.__init__(self, index=index, name=name)
        self.nuclei = []
        for n in nuclei:
            n['coordinate'] = np.array(n['coordinate'])
            self.nuclei.append(particle.Nucleus(**n))
        self.num_nuclei = len(nuclei)
        self.density = e_density_matrix.density
