#ifndef _python_eigen_conversions_h_
#define _python_eigen_conversions_h_

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "numpy/arrayobject.h"

#include <Eigen/Dense>
#include <vector>
#include <iostream>

namespace conversion
{
//Reads a multiindex into an Eigen::Matrix.
//multiindex_obj: Points to a List of two numpy.ndarrays of length 3
Eigen::Matrix<int, 2, 3> read_multiindex(PyObject* multiindex_obj);

//Reads a vector of length 3 into an Eigen::Vector3d.
//array_obj: Points to an np.ndarray of length 3
Eigen::Vector3d read_vector3d(PyArrayObject* array_obj);

//Reads a 3D tensor into a std..vector<Eigen::MatrixXd>. tensor[i, j, k] == return_value[i](j, k).
//tensor_obj: Points to a 3-dimensional np.ndarray containing tensor coefficients
std::vector<Eigen::MatrixXd> read_tensor(PyArrayObject* tensor_obj);

//Reads a 2D np.ndarray into an Eigen::MatrixXd.
//matrix_obj: Points to a 2-dimensional np.ndarray containing tensor coefficients
Eigen::MatrixXd read_matrix(PyArrayObject* matrix_obj);

//Reads a 1D np.ndarray into an Eigen::VectorXd.
//matrix_obj: Points to a 1-dimensional np.ndarray containing tensor coefficients
Eigen::VectorXi read_vector(PyArrayObject* vector_obj);

//Creates, from an Eigen::MatrixXd, a 2-dimensional np.ndarray of the same shape.
//matrix: the matrix to use
PyObject* eigen_matrix_to_numpy(const Eigen::MatrixXd &matrix);

//Reads the tensor template. template[i, j][k][l] = return_value(i, j)(k, l).
//template_obj: 2-dimensional np.ndarray of lists (length 2) of 1-dimensional np.ndarrays (length 3).
Eigen::Matrix<Eigen::Matrix<int, 2, 3>, Eigen::Dynamic, Eigen::Dynamic> read_tensor_template(PyArrayObject* template_obj);

//Must be called at least once before using any of the functions
void init();

}

#endif