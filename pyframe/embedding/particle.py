from __future__ import annotations

import numpy as np
import qcelemental
from typing import Optional
from pyframe.embedding import polytensor, tensor_tools, constants, interaction_tensor
from scipy.optimize import root


class Particle:
    """A Particle object represents a particle of a Classical- or Quantum-subsystem: atoms in the environment
    (Classical subsystem), virtual-particles in the environment (Classical subsystem) that represents sites that do not
    physically exist (e.g., a representation of a covalent bond), and nuclei in the core (Quantum subsystem).

    Args:
        index: The particle's index.
        mass: The mass of the particle.
        coordinate: The cartesian coordinates of the particle.
    """

    def __init__(self,
                 index: int,
                 coordinate: np.ndarray,
                 mass: Optional[float] = None
                 ):
        self.index = index
        self._mass = mass
        self.coordinate = coordinate
        self.particle_variables = constants.values


class Atom(Particle):
    """An Atom object represents an atom or super-atom in the environment and inherits from the Particle class. A
    super-atom represents several atoms as one atom.

    Args:
        index: The atom's index.
        element: The element of the atom.
        mass: The mass of the atom.
        coordinate: The cartesian coordinates of the atom.
        multipoles: Dictionary of multipoles of the atoms under 'elements' up-to the order under 'order'.
        polarizabilities: Dictionary of polarizabilities of the atoms under 'elements' of the order under 'order'.
        exclusions:
        vdw: Dictionary containing the 'vdw_method', 'lj_sigma', and 'lj_epsilon'.
    """

    def __init__(self,
                 index: int,
                 coordinate: np.array,
                 induced_dipole: Optional[np.ndarray] = None,
                 name: Optional[str] = None,
                 exclusions: Optional[list] = None,
                 mass: Optional[float] = None,
                 element: Optional[str] = None,
                 repulsion: Optional[dict] = None,
                 dispersion: Optional[dict] = None,
                 multipoles: Optional[dict] = None,
                 polarizabilities: Optional[dict] = None,
                 ):
        Particle.__init__(self, index=index, mass=mass, coordinate=coordinate)
        if exclusions is not None and not isinstance(exclusions, list):
            raise ValueError("Exclusions must be a list.")
        self.name = name
        self.induced_dipole = induced_dipole
        if exclusions is not None and isinstance(exclusions, list):
            self.exclusions = tuple(exclusions)
        self._element = element
        if repulsion is not None:
            self.repulsion = repulsion.get('method', None)
            self.rep_parameters = repulsion.get('parameters', None)
        if dispersion is not None:
            self.dispersion = dispersion.get('method', None)
            self.disp_parameters = dispersion.get('parameters', None)
        if multipoles is not None:
            m_elements = np.array(multipoles.get('elements', None))
            if multipoles.get('order', None) is None:
                self.multipole_order = multipole_len_to_order(len(m_elements))
            else:
                self.multipole_order = multipoles.get('order', None)
            if self.multipole_order > 1:
                counter = 4
                for n in range(2, self.multipole_order + 1):
                    i = (n + 1) * (n + 2) // 2
                    m_elements[counter:(i + counter)] = tensor_tools.detrace(m_elements[counter:(i + counter)],
                                                                             self.particle_variables.factorials,
                                                                             self.particle_variables.double_factorials,
                                                                             self.particle_variables.trinomials)
                    counter += i
            self.multipoles = polytensor.FirstDegreePolytensor(self.multipole_order,
                                                               tensor_data=m_elements,
                                                               data_type=np.float64)
            self.degeneracy_tensor = self.particle_variables.degeneracies.truncate_tensor(order=self.multipole_order)
            self.multipoles_with_degeneracy = polytensor.FirstDegreePolytensor. \
                multiply_elementwise(self.multipoles, self.degeneracy_tensor)
            self.taylor_coefficients = polytensor.FirstDegreePolytensor(self.multipole_order)
            for i in range(self.multipole_order + 1):
                self.taylor_coefficients.write_to_data_block_wise(np.full((i + 1) * (i + 2) // 2, (-1) ** i
                                                                          / self.particle_variables.factorials[i]))
        if polarizabilities is not None:
            p_elements = np.array(polarizabilities.get('elements', None))
            self.polarizability = p_elements
            self.polarizability_order = np.array(polarizabilities.get('order', None))
        else:
            self.polarizability = np.array([], dtype=np.float64)

    def potential(self,
                  coordinate: np.ndarray,
                  pot_derivative_order: Optional[int] = 0,
                  origin_derivative_order: Optional[int] = 0,
                  coord_multipole_order: Optional[int] = 0
                  ) -> float | np.ndarray:
        """Calculates the electrostatic potential and its derivatives of a multipole. It is based on a vector-vector/
        vector-matrix product M*T/M@T, where M are the compressed multipoles including the degeneracy factors, and T is
        a vector or matrix corresponding to the derivatives of 1/|r_coord - r_atom| wrt x, y, and z components of
        r_coord or r_atom.

        Args:
            coordinate: Coordinates at which the potential is evaluated.
            pot_derivative_order: Order of the derivative of the potential.
            origin_derivative_order: Order of derivative with respect to the origin of the potential.
            coord_multipole_order: Multipole order at coordinate.
        Returns:
            Electrostatic potential or its derivative of the particle at coordinates. If coord_multipole_order is given,
            the derivatives with respect to the charge or multipole at coordinate are included.
        """
        if coord_multipole_order == 0:
            is_potential = True
        else:
            is_potential = False
        t_tensor = interaction_tensor.compute_t_tensor(r_a=self.coordinate,
                                                       r_b=coordinate,
                                                       rank_a=self.multipole_order,
                                                       rank_b=pot_derivative_order
                                                       + origin_derivative_order
                                                       + coord_multipole_order,
                                                       is_potential=is_potential,
                                                       start_rank_a=0,
                                                       start_rank_b=pot_derivative_order
                                                       + origin_derivative_order)
        t_tensor.data = t_tensor.data * (-1) ** origin_derivative_order
        return polytensor.FirstDegreePolytensor. \
            multiply_elementwise(self.multipoles_with_degeneracy, self.taylor_coefficients). \
            multiply_first_degree_second_degree(t_tensor).data


def multipole_len_to_order(multipoles) -> int:
    """Calculates the maximum multipole order of an array of compressed multipoles ordered in ascending order.

    Args:
        multipoles: Array of compressed multipoles.

    Returns:
        Multipole order.
    """
    if not isinstance(multipoles, int) or multipoles < 0:
        raise ValueError("Input must be a non-negative integer.")

    def equation(t):
        return ((t + 1) * (t + 2) * (t + 3)) / 6 - multipoles

    solution = root(equation, np.array([0]))
    return round(solution.x[0])


class Nucleus(Particle):
    """A Nucleus object represents a nucleus in the core and inherits from the Particle class.

    Args:
        index: The nucleus' index.
        element: The element of the nucleus.
        mass: The mass of the nucleus.
        charge: The charge of the nucleus.
        coordinate: The cartesian coordinates of the nucleus.
        vdw: Dictionary containing the 'vdw_method', 'lj_sigma', and 'lj_epsilon'.
    """

    def __init__(self,
                 index: int,
                 coordinate: np.ndarray,
                 charge: float,
                 name: Optional[str] = None,
                 mass: Optional[float] = None,
                 element: Optional[str] = None,
                 repulsion: Optional[dict] = None,
                 dispersion: Optional[dict] = None,
                 ):
        Particle.__init__(self, index=index, mass=mass, coordinate=coordinate)
        self.name = name
        self.charge = np.array([charge], dtype=np.float64)
        if element is not None:
            self._element = element
            if self.charge != self.element_to_charge():
                raise ValueError("Element does not match Charge.")
        else:
            self._element = self.charge_to_element()
        if repulsion is not None:
            self.repulsion = repulsion.get('method', None)
            self.rep_parameters = repulsion.get('parameters', None)
        if dispersion is not None:
            self.dispersion = dispersion.get('method', None)
            self.disp_parameters = dispersion.get('parameters', None)

    def charge_to_element(self) -> str:
        """Identifies the element string from its corresponding nuclear charge.

        Returns:
            Element string.
        """
        return qcelemental.periodictable.to_element(self.charge)

    def element_to_charge(self) -> float:
        """Identifies the nuclear charge from its corresponding element string.

        Returns:
            Nuclear charge.
        """
        return qcelemental.periodictable.to_atomic_number(self._element)

    def potential(self,
                  coordinate: np.ndarray,
                  pot_derivative_order: Optional[int] = 0,
                  origin_derivative_order: Optional[int] = 0,
                  coord_multipole_order: Optional[int] = 0
                  ) -> float | np.ndarray:
        """Calculates the electrostatic potential and its derivatives of a Nucleus based on a scalar-vector product Z*T,
         where Z is the charge of the nucleus, and T is a vector or matrix corresponding to the derivatives of
         1/|r_coord - r_nucleus| wrt x, y, and z components of r_coord or r_nucleus.

        Args:
            coordinate: Coordinates at which the potential is evaluated.
            pot_derivative_order: Order of the derivative of the potential.
            origin_derivative_order: Order of derivative with respect to the origin of the potential.
            coord_multipole_order: Multipole order at coordinate.
        Returns:
            Electrostatic potential or its derivative of the Nucleus at coordinates. If coord_multipole_order is given,
            the derivatives with respect to the charge or multipole at coordinate are included.
        """
        if coord_multipole_order == 0:
            is_potential = True
        else:
            is_potential = False
        t_tensor = interaction_tensor.compute_t_tensor(r_a=self.coordinate,
                                                       r_b=coordinate,
                                                       rank_a=0,
                                                       rank_b=pot_derivative_order
                                                       + origin_derivative_order
                                                       + coord_multipole_order,
                                                       is_potential=is_potential,
                                                       start_rank_a=0,
                                                       start_rank_b=pot_derivative_order
                                                       + origin_derivative_order)
        t_tensor.data = t_tensor.data * (-1) ** origin_derivative_order
        return polytensor.FirstDegreePolytensor(rank=0,
                                                tensor_data=self.charge,
                                                data_type=np.float64). \
            multiply_first_degree_second_degree(t_tensor).data


class VirtualParticle(Particle):
    """A SuperAtom object represents super-atoms in the environment and inherits from the Particle class. Super-atoms
    represent several atoms (a molecule or several molecules) in the environment.

    Args:
        index: The virtual particle's index.
        coordinate: The cartesian coordinates of the virtual particle.
    """

    def __init__(self,
                 index: int,
                 coordinate: np.array
                 ):
        Particle.__init__(self, index=index, coordinate=coordinate)
