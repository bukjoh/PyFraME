# Copyright (C) 2017-2021  Jógvan Magnus Haugaard Olsen and Peter Reinholdt
#
# This file is part of PyFraME.
#
# PyFraME is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyFraME is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyFraME.  If not, see <https://www.gnu.org/licenses/>.
#

""" The module 'simulation_box' contains the class 'SimulationBox', which defines the system simulation box."""

import numpy as np


__all__ = ['SimulationBox']


class SimulationBox:
    """Define the system simulation box."""
    def __init__(self,
                 lengths=None,
                 angles=None):
        if lengths is not None:
            self.lengths = lengths
        else:
            self.lengths = np.zeros(3)
        if angles is not None:
            self.angles = angles
        else:
            self.angles = np.zeros(3)
        if lengths is not None and angles is not None:
            self.box = self.create_box()
        else:
            self.box = np.zeros([3, 3])

    def create_box(self):
        angles_rad = np.radians(self.angles)
        # Matrix elements
        a1 = self.lengths[0]
        b1 = self.lengths[1] * np.cos(angles_rad[2])
        b2 = self.lengths[1] * np.sin(angles_rad[2])
        c1 = self.lengths[2] * np.cos(angles_rad[1])
        c2 = self.lengths[2] * (np.cos(angles_rad[0]) - np.cos(angles_rad[1]) * np.cos(angles_rad[2])) / np.sin(angles_rad[2])
        c3 = np.sqrt(self.lengths[2] ** 2 - c1 ** 2 - c2 ** 2)
        # Constructing the matrix
        self.box = np.array([[a1, 0, 0],
                            [b1, b2, 0],
                            [c1, c2, c3]])
        # Set very small values close to zero explicitly to zero
        self.box[np.isclose(self.box, 0, atol=1e-12)] = 0
        return self.box
