"""Tests PyFraME.embedding.density_matrix.py"""
from pyframe.embedding import density_matrix
import numpy as np
import pytest


class TestDensityMatrix:
 def test_init_with_numpy_array(self,
                                water_density
                                ):
  dens_mat = density_matrix.DensityMatrix(density=water_density)
  assert np.array_equal(dens_mat.density, water_density)

 def test_init_with_list(self):
  density_list = [[1.0, 0.0], [0.0, 1.0]]
  dens_mat = density_matrix.DensityMatrix(density=density_list)
  assert np.array_equal(dens_mat.density, np.array(density_list))

 def test_init_with_invalid_density_type(self):
  with pytest.raises(ValueError, match="Density must be a numpy array or a list"):
   density_matrix.DensityMatrix(density="invalid_type")
