from __future__ import annotations

import numpy as np


class DensityMatrix:
    """A DensityMatrices object represents a matrix that represents an electron density matrix.
    Args:
        density: Two dimensional tensor representing the measure of the probability of the object being present at an
        infinitesimal element of space surrounding any given point.
    """
    def __init__(self,
                 density: np.ndarray | list
                 ):
        if isinstance(density, list):
            density = np.array(density)
        self.density = density

