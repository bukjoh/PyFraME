from __future__ import annotations

import numpy as np


class DensityMatrix:
    """A DensityMatrix object is a matrix that represents a density.
    Args:
        density: Two dimensional tensor representing the measure of the probability of the object being present at an
        infinitesimal element of space surrounding any given point.
    """
    def __init__(self,
                 density: np.ndarray | list
                 ):
        if not isinstance(density, (np.ndarray, list)):
            raise ValueError("Density must be a numpy array or a list")
        if isinstance(density, list):
            density = np.array(density)
        self.density = density

