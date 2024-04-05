// Some core functionality implemented in C++
// Multiprocessing using OpenMP is partially implemented.
// Caution is advised regarding the validity of inputs, as some common errors are checked, but not all.
// Undefined states may result from invalid inputs.

// Some issues with the numpy API:
// - Problems when calling numpy from different .cpp files. Therefore, numpy functions
//   should be called only from this file
// - Problems when passing anything other than numpy arrays or lists of numpy arrays.

// Possible performance improvements: 
// - Use Eigen::Map to map numpy.ndarrays directly to Eigen Matrices instead of copying values
// - Use row-major matrices to better pass data between Eigen and Numpy


#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "numpy/arrayobject.h"

#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <unordered_set>

#include "computation.h"
#include "global.h"  

// Reads a multiindex into an Eigen::Matrix.
// multiindex_obj: Points to a List of two np.ndarrays of length 3
Eigen::Matrix<int, 2, 3> read_multiindex(PyObject *multiindex_obj)
{
    // Check if multiindex is a list
    if (!PyList_Check(multiindex_obj))
    {
        PyErr_SetString(PyExc_TypeError, "multiindex must be a list of NumPy arrays");
        return Eigen::Matrix<int, 2, 3>();
    }

    Eigen::Matrix<int, 2, 3> multiindex;

    // Extract data from the first NumPy array in multiindex, holding values ax, ay, az
    PyObject *array1 = PyList_GetItem(multiindex_obj, 0);
    if (!PyArray_Check(array1) || PyArray_SIZE((PyArrayObject *)array1) != 3)
    {
        PyErr_SetString(PyExc_ValueError, "Each array in multiindex must be a NumPy array of length 3");
        return Eigen::Matrix<int, 2, 3>();
    }

    PyArray_Descr *descr1 = PyArray_DESCR((PyArrayObject *)array1);
    if (descr1 == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to get array descriptor");
        return Eigen::Matrix<int, 2, 3>();
    }
    int type_num1 = descr1->type_num;

    if (type_num1 == NPY_INT32)
    {
        int32_t *multiindex_data1 = static_cast<int32_t *>(PyArray_DATA((PyArrayObject *)array1));
        multiindex(0, 0) = multiindex_data1[0];
        multiindex(0, 1) = multiindex_data1[1];
        multiindex(0, 2) = multiindex_data1[2];
    }
    else if (type_num1 == NPY_INT64)
    {
        int64_t *multiindex_data1 = static_cast<int64_t *>(PyArray_DATA((PyArrayObject *)array1));
        multiindex(0, 0) = (int)multiindex_data1[0];
        multiindex(0, 1) = (int)multiindex_data1[1];
        multiindex(0, 2) = (int)multiindex_data1[2];
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Unsupported data type");
        return Eigen::Matrix<int, 2, 3>();
    }

    // Extract data from the second NumPy array in multiindex, holding values bx, by, bz
    PyObject *array2 = PyList_GetItem(multiindex_obj, 1);
    if (!PyArray_Check(array2) || PyArray_SIZE((PyArrayObject *)array2) != 3)
    {
        PyErr_SetString(PyExc_ValueError, "Each array in multiindex must be a NumPy array of length 3");
        return Eigen::Matrix<int, 2, 3>();
    }

    PyArray_Descr *descr2 = PyArray_DESCR((PyArrayObject *)array2);
    if (descr2 == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to get array descriptor");
        return Eigen::Matrix<int, 2, 3>();
    }

    int type_num2 = descr2->type_num;

    if (type_num2 == NPY_INT32)
    {
        int32_t *multiindex_data2 = static_cast<int32_t *>(PyArray_DATA((PyArrayObject *)array2));
        multiindex(1, 0) = multiindex_data2[0];
        multiindex(1, 1) = multiindex_data2[1];
        multiindex(1, 2) = multiindex_data2[2];
    }
    else if (type_num2 == NPY_INT64)
    {
        int64_t *multiindex_data2 = static_cast<int64_t *>(PyArray_DATA((PyArrayObject *)array2));
        multiindex(1, 0) = (int)multiindex_data2[0];
        multiindex(1, 1) = (int)multiindex_data2[1];
        multiindex(1, 2) = (int)multiindex_data2[2];
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Unsupported data type");
        return Eigen::Matrix<int, 2, 3>();
    }
    return multiindex;
}

// Reads a vector of length 3 into an Eigen::Vector3d.
// array_obj: Points to an np.ndarray of length 3
Eigen::Vector3d read_vector3d(PyObject *array_obj)
{
    if (!PyArray_Check(array_obj))
    {
        PyErr_SetString(PyExc_TypeError, "vector must be NumPy array");
        return Eigen::Vector3d();
    }
    if (PyArray_NDIM((PyArrayObject *)array_obj) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "vector must be a one-dimensional array");
        return Eigen::Vector3d();
    }
    Eigen::Vector3d vect;
    for (int i = 0; i < 3; ++i)
    {
        vect(i) = *(double *)PyArray_GETPTR1((PyArrayObject *)array_obj, i);
    }
    return vect;
}

// Reads a 3D tensor into a std..vector<Eigen::MatrixXd>. tensor[i, j, k] == return_value[i](j, k).
// tensor_obj: Points to a 3-dimensional np.ndarray contining tensor coefficients
std::vector<Eigen::MatrixXd> read_tensor(PyArrayObject *tensor_obj)
{
    // Check if tensor is NumPy array
    if (!PyArray_Check(tensor_obj))
    {
        PyErr_SetString(PyExc_TypeError, "tensor must be NumPy array");
        return std::vector<Eigen::MatrixXd>();
    }
    if (PyArray_NDIM(tensor_obj) != 3)
    {
        PyErr_SetString(PyExc_ValueError, "tensor must be a three-dimensional array");
        return std::vector<Eigen::MatrixXd>();
    }
    npy_intp t = PyArray_DIM(tensor_obj, 0), u = PyArray_DIM(tensor_obj, 1), v = PyArray_DIM(tensor_obj, 2);
    std::vector<Eigen::MatrixXd> tensor;
    for (npy_intp i = 0; i < t; ++i)
    {
        Eigen::MatrixXd matrix(u, v);
        for (int j = 0; j < u; ++j)
        {
            for (int k = 0; k < v; ++k)
            {
                matrix(j, k) = *(double *)PyArray_GETPTR3((PyArrayObject *)tensor_obj, i, j, k);
            }
        }
        tensor.push_back(matrix);
    }
    return tensor;
}

// Reads a 2D np.ndarray into an Eigen::MatrixXd.
// matrix_obj: Points to a 2-dimensional np.ndarray containing tensor coefficients
Eigen::MatrixXd read_matrix_d(PyArrayObject *matrix_obj)
{
    if (!PyArray_Check(matrix_obj) || PyArray_NDIM(matrix_obj) != 2)
    {
        PyErr_SetString(PyExc_TypeError, "matrix must be a two-dimensional NumPy array");
        return Eigen::MatrixXd();
    }
    npy_intp u = PyArray_DIM(matrix_obj, 0), v = PyArray_DIM(matrix_obj, 1);
    Eigen::MatrixXd matrix(u, v);
    for (int j = 0; j < u; ++j)
    {
        for (int k = 0; k < v; ++k)
        {
            matrix(j, k) = *(double *)PyArray_GETPTR2((PyArrayObject *)matrix_obj, j, k);
        }
    }
    return matrix;
}


// Reads a 2D np.ndarray into an Eigen::MatrixXi.
// matrix_obj: Points to a 2-dimensional np.ndarray containing tensor coefficients
Eigen::MatrixXi read_matrix_i(PyArrayObject *matrix_obj)
{
    if (!PyArray_Check(matrix_obj) || PyArray_NDIM(matrix_obj) != 2)
    {
        PyErr_SetString(PyExc_TypeError, "matrix must be a two-dimensional NumPy array");
        return Eigen::MatrixXi();
    }
    npy_intp u = PyArray_DIM(matrix_obj, 0), v = PyArray_DIM(matrix_obj, 1);
    Eigen::MatrixXi matrix(u, v);
    for (int j = 0; j < u; ++j)
    {
        for (int k = 0; k < v; ++k)
        {
            matrix(j, k) = *(int *)PyArray_GETPTR2((PyArrayObject *)matrix_obj, j, k);
        }
    }
    return matrix;
}


// Reads a 1D np.ndarray into an Eigen::VectorXi.
Eigen::VectorXi read_vector(PyObject *vector_obj)
{
    if (!PyArray_Check(vector_obj) || PyArray_NDIM((PyArrayObject *)vector_obj) != 1)
    {
        PyErr_SetString(PyExc_TypeError, "vector must be a one-dimensional NumPy array");
        return Eigen::VectorXi();
    }
    npy_intp u = PyArray_DIM((PyArrayObject *)vector_obj, 0);
    Eigen::VectorXi vector(u);
    for (int j = 0; j < u; ++j)
    {
        vector(j) = (int)*(long *)PyArray_GETPTR1((PyArrayObject *)vector_obj, j);
    }
    return vector;
}

// Reads a 1D np.ndarray into an Eigen::VectorXl.
Eigen::VectorXd read_vector_d(PyObject *vector_obj)
{
    if (!PyArray_Check(vector_obj) || PyArray_NDIM((PyArrayObject *)vector_obj) != 1)
    {
        PyErr_SetString(PyExc_TypeError, "vector must be a one-dimensional NumPy array");
        return Eigen::VectorXd();
    }
    npy_intp u = PyArray_DIM((PyArrayObject *)vector_obj, 0);
    Eigen::VectorXd vector(u);
    for (int j = 0; j < u; ++j)
    {
        vector(j) = *(double *)PyArray_GETPTR1((PyArrayObject *)vector_obj, j);
    }
    return vector;
}

std::vector<Eigen::VectorXd> read_multipoles(PyObject *multipoles_obj)
{
    // Check if multipoles_obj is a list
    if (!PyList_Check(multipoles_obj))
    {
        PyErr_SetString(PyExc_TypeError, "Multipoles must be a list of NumPy arrays");
        return std::vector<Eigen::VectorXd>();
    }

    std::vector<Eigen::VectorXd> multipoles;
    Py_ssize_t num_multipoles = PyList_Size(multipoles_obj);

    for (Py_ssize_t i = 0; i < num_multipoles; ++i) {
        PyObject *numpy_array = PyList_GetItem(multipoles_obj, i);
        PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(numpy_array);

        int ndim = PyArray_NDIM(arr);
        if (ndim != 1) {
            PyErr_SetString(PyExc_TypeError, "Expected 1-dimensional arrays");
            return std::vector<Eigen::VectorXd>(); // Return or throw, don't mix
        }

        int array_size = PyArray_DIM(arr, 0);
        double *ptr = reinterpret_cast<double *>(PyArray_DATA(arr));
        Eigen::Map<Eigen::VectorXd> vector(ptr, array_size);
        multipoles.push_back(vector);
    }

    return multipoles;
}


// Creates, from an Eigen::MatrixXd, a 2-dimensional np.ndarray of the same shape.
// matrix: the matrix to use
PyObject *eigen_matrix_to_numpy(const Eigen::MatrixXd &matrix)
{
    int rows = (int)matrix.rows();
    int cols = (int)matrix.cols();

    npy_intp dims[] = {rows, cols};
    PyObject *numpyArray = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (numpyArray == NULL)
    {
        return NULL; // Memory allocation failed
    }

    // Copy data from Eigen matrix to NumPy array, but first converting to row-major order
    double *numpyData = static_cast<double *>(PyArray_DATA((PyArrayObject *)numpyArray));
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_rm = matrix;
    const double *eigenData = matrix_rm.data();
    std::copy(eigenData, eigenData + rows * cols, numpyData);

    return numpyArray;
}

// Reads the tensor template. template[i, j][k][l] = return_value(i, j)(k, l).
// template_obj: 2-dimensional np.ndarray of lists (length 2) of 1-dimensional np.ndarrays (length 3).
Eigen::Matrix<Eigen::Matrix<int, 2, 3>, Eigen::Dynamic, Eigen::Dynamic> read_tensor_template(PyArrayObject *template_obj)
{
    // Check if the input is a valid 2D NumPy array
    if (!PyArray_Check(template_obj) || PyArray_NDIM(template_obj) != 2)
    {
        PyErr_SetString(PyExc_TypeError, "tensor_template must be a 2D NumPy array");
        return Eigen::Matrix<Eigen::Matrix<int, 2, 3>, Eigen::Dynamic, Eigen::Dynamic>(0, 0);
    }
    npy_intp *dims = PyArray_DIMS(template_obj);
    npy_intp rows = dims[0];
    npy_intp cols = dims[1];

    Eigen::Matrix<Eigen::Matrix<int, 2, 3>, Eigen::Dynamic, Eigen::Dynamic> tensor_template(rows, cols);

    for (npy_intp i = 0; i < rows; ++i)
    {
        for (npy_intp j = 0; j < cols; ++j)
        {
            PyObject *element = PyArray_GETITEM(template_obj, (char *)PyArray_GETPTR2(template_obj, i, j));
            tensor_template(i, j) = read_multiindex(element);
        }
    }
    return tensor_template;
}

// Computes an interaction tensor element from a given multiindex, r_ab == r_b - r_a, and tensor_coefficients
// args: [multiindex, r_ab, tensor_coefficients]
static PyObject* compute_interaction_tensor_element(PyObject* self, PyObject* args) {
    PyObject *multiindex_obj, *r_ab_array;

    if (!PyArg_ParseTuple(args, "OO", &multiindex_obj, &r_ab_array))
    {
        return NULL;
    }

    Eigen::Matrix<int, 2, 3> multiindex = read_multiindex(multiindex_obj);
    Eigen::Vector3d r_ab = read_vector3d(r_ab_array);

    double result = computation::compute_interaction_tensor_element(multiindex, r_ab, global::tensor_coefficients);

    return Py_BuildValue("d", result);
}


// Computes the t_tensor for the interaction of two atoms in a specified range of indices.
// args: [r_a, r_b, rank_a, rank_b, start_rank_a, start_rank_b, is_potential (true: potential, false: interaction)]
static PyObject* compute_t_tensor(PyObject* self, PyObject* args) {
    PyObject *r_a_obj, *r_b_obj;
    PyObject* ranks_obj;

    if (!PyArg_ParseTuple(args, "OOO", &r_a_obj, &r_b_obj,
                          &ranks_obj))
    {
        return NULL;
    }
    Eigen::VectorXi ranks = read_vector(ranks_obj);
    int rank_a = ranks(0);
    int rank_b = ranks(1);
    int start_rank_a = ranks(2);
    int start_rank_b = ranks(3);
    bool is_potential = (bool)ranks(4);

    Eigen::Vector3d r_a = read_vector3d(r_a_obj);
    Eigen::Vector3d r_b = read_vector3d(r_b_obj);
    if (r_a[0] == r_b[0] && r_a[1] == r_b[1] && r_a[2] == r_b[2])
    {
        PyErr_SetString(PyExc_ValueError, "r_a and r_b cannot be equal.");
        return eigen_matrix_to_numpy(Eigen::MatrixXd(0, 0));
    }
    // TODO: Implement a mechanism ensuring sufficient array size
    // or make sure the first call to this function has the maximum required size (which seems to be the case?)

    Eigen::MatrixXd t_tensor = computation::compute_t_tensor(r_b - r_a,
                                is_potential ? global::tensor_template_potential : global::tensor_template_interaction,
                                (int)rank_a, (int)rank_b, (int)start_rank_a, (int)start_rank_b);

    return eigen_matrix_to_numpy(t_tensor);
}

// Sets the global tensor coefficients and templates
// args: [tensor_coefficients, tensor_template_interaction, tensor_template_potential, rank, max_order]
static PyObject* set_tensor_coefficients(PyObject* self, PyObject* args) {
    PyObject *tensor_coefficients_obj, *tensor_template_interaction_obj, *tensor_template_potential_obj;
    int rank, max_order;
    if (!PyArg_ParseTuple(args, "OOOii", &tensor_coefficients_obj, &tensor_template_interaction_obj,
                          &tensor_template_potential_obj, &rank, &max_order)) {
        return NULL;
    }
    if(rank > global::rank) {
        // Check if the input is a valid 2D NumPy array
        if (!PyArray_Check(tensor_template_interaction_obj) || PyArray_NDIM((PyArrayObject*)tensor_template_interaction_obj) != 2) {
            PyErr_SetString(PyExc_TypeError, "tensor_template must be a 2D NumPy array");
            Py_RETURN_NONE;
        }
        global::tensor_template_interaction = read_tensor_template((PyArrayObject *)tensor_template_interaction_obj);
        global::tensor_template_potential = read_tensor_template((PyArrayObject *)tensor_template_potential_obj);
        global::rank = (long)rank;
    }
    if(max_order > global::max_order) {
        global::tensor_coefficients = read_tensor((PyArrayObject *)tensor_coefficients_obj);
        global::max_order = (long)max_order;
    }
    Py_RETURN_NONE;
}

// Sets the global coordinates, indices and exclusions.
// args: [coords, indices, exclusions];
static PyObject* set_coords_idxs_exlcs(PyObject* self, PyObject* args) {
    PyObject *coords_obj, *indices_obj, *exclusions_obj;

    if (!PyArg_ParseTuple(args, "OOO", &coords_obj, &indices_obj, &exclusions_obj)) {
        return NULL;
    }
    Eigen::MatrixXd coords = read_matrix_d((PyArrayObject *)coords_obj);
    global::coordinates = std::vector<Eigen::Vector3d>();
    for(int i = 0; i < coords.rows(); i++) {
        Eigen::Vector3d coord;
        coord << coords(i, 0), coords(i, 1), coords(i, 2);
        global::coordinates.push_back(coord);
    }
    global::indices = read_vector(indices_obj);

    if (!PyList_Check(exclusions_obj)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a Python list");
        return NULL;
    }
    Py_ssize_t outerSize = PyList_Size(exclusions_obj);
    std::vector<std::unordered_set<int>> exclusions;
    for (Py_ssize_t i = 0; i < outerSize; ++i) {
        PyObject* inner_tuple = PyList_GetItem(exclusions_obj, i);
        if (!PyTuple_Check(inner_tuple)) {
            PyErr_SetString(PyExc_TypeError, "Inner items must be Python tuples");
            return NULL;
        }
        std::unordered_set<int> inner_set;
        Py_ssize_t inner_size = PyTuple_Size(inner_tuple);
        for (Py_ssize_t j = 0; j < inner_size; ++j) {
            PyObject* item = PyTuple_GetItem(inner_tuple, j);
            if (!PyLong_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "Inner items must be Python integers");
                return NULL;
            }
            int intValue = PyLong_AsLong(item);
            inner_set.insert(intValue);
        }
        exclusions.push_back(inner_set);
    }
    global::exclusions = exclusions;
    Py_RETURN_NONE;
}

// Sets the global atom coordinates,nuclei_coordinates and nuclei_charges.
// args: [atom_coords, nuclei_charges, nuclei_coords];
static PyObject* set_coords_nuc_coords_charges(PyObject* self, PyObject* args) {
    PyObject *atom_coords_obj, *nuc_charges_obj, *nuc_coords_obj;
    if (!PyArg_ParseTuple(args, "OOO", &atom_coords_obj, &nuc_charges_obj, &nuc_coords_obj)) {
        return NULL;
    }
    Eigen::MatrixXd coords = read_matrix_d((PyArrayObject *)atom_coords_obj);
    global::coordinates = std::vector<Eigen::Vector3d>();
    for(int i = 0; i < coords.rows(); i++) {
        Eigen::Vector3d coord;
        coord << coords(i, 0), coords(i, 1), coords(i, 2);
        global::coordinates.push_back(coord);
    }
    Eigen::MatrixXd nuc_coords = read_matrix_d((PyArrayObject *)nuc_coords_obj);
    global::nuclei_coordinates = std::vector<Eigen::Vector3d>();
    for(int i = 0; i < nuc_coords.rows(); i++) {
        Eigen::Vector3d coord;
        coord << nuc_coords(i, 0), nuc_coords(i, 1), nuc_coords(i, 2);
        global::nuclei_coordinates.push_back(coord);
    }
    global::nuclei_charges = read_vector_d(nuc_charges_obj);
    Py_RETURN_NONE;
}

// Sets the old induced dipoles for the calculation of the induced dipole fields
// args: [old_ind_dipoles]
static PyObject *set_old_ind_dipoles(PyObject *self, PyObject *args)
{
    PyObject *old_ind_dipoles_obj;
    if (!PyArg_ParseTuple(args, "O", &old_ind_dipoles_obj))
    {
        return NULL;
    }
    global::old_ind_dipoles = read_matrix_d((PyArrayObject *)old_ind_dipoles_obj);
    Py_RETURN_NONE;
}

// Sets the multipoles with degeneracy and taylor coefficients and multipole orders for the calculation of the multipole fields.
// args: [multipoles, multipole_orders]
static PyObject *set_multipoles_multipoles_order(PyObject *self, PyObject *args)
{
    PyObject *multipoles_obj, *multipole_orders_obj;
    if (!PyArg_ParseTuple(args, "OO", &multipoles_obj, &multipole_orders_obj)) {
        return NULL;
    }
    // Set MultipoleOrder
    global::multipole_orders = read_vector(multipole_orders_obj);
    // Set Multipoles
    global::multipoles = read_multipoles(multipoles_obj);
    Py_RETURN_NONE;
}

//Calculates the induced dipoles from start to end
//args: [start, end] (numpy.ndarray with start and end as entries)
static PyObject* ind_dipoles_fields(PyObject* self, PyObject* args) {
    PyObject *start_end_obj;
    if (!PyArg_ParseTuple(args, "O", &start_end_obj)) {
        return NULL;
    }
    int start = (int)read_vector(start_end_obj)(0);
    int end = (int)read_vector(start_end_obj)(1);
    return (PyObject *)eigen_matrix_to_numpy(computation::ind_dipoles_field(start, end));
}

//Calculates the multipole fields for atom at index i
//args: [[i]] (numpy.ndarray with i as only entry)
static PyObject* multipole_fields(PyObject* self, PyObject* args) {
    PyObject *i_obj;
    if (!PyArg_ParseTuple(args, "O", &i_obj)) {
        return NULL;
    }
    int i = (int)read_vector(i_obj)(0);
    return (PyObject *)eigen_matrix_to_numpy(computation::multipole_field(i));
}

//Calculates the nuclei fields on coordinates
//args: [start, end] (numpy.ndarray with start and end as entries)
static PyObject* nuclei_fields(PyObject* self, PyObject* args) {
    PyObject *start_end_obj;
    if (!PyArg_ParseTuple(args, "O", &start_end_obj)) {
        return NULL;
    }
    int start = (int)read_vector(start_end_obj)(0);
    int end = (int)read_vector(start_end_obj)(1);
    return (PyObject *)eigen_matrix_to_numpy(computation::nuclei_fields(start, end));
}

// Function to convert std::vector<Eigen::MatrixXd> to NumPy array
PyObject* std_vec_of_eigen_matrixXd_to_numpy(const std::vector<Eigen::MatrixXd>& matrices) {
    if (matrices.empty()) {
        PyErr_SetString(PyExc_ValueError, "Input vector is empty");
        return nullptr;
    }

    // Determine the shape of the resulting NumPy array
    npy_intp shape[] = {static_cast<npy_intp>(matrices.size()), matrices[0].rows(), matrices[0].cols()};

    // Create a NumPy array and fill it with data from the vector of Eigen matrices
    PyObject* numpyArray = PyArray_SimpleNew(3, shape, NPY_DOUBLE);
    double* data = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(numpyArray)));

    for (const auto& matrix : matrices) {
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                *data++ = matrix(i, j); // Copy data in row-major order
            }
        }
    }

    return numpyArray;
}

//Calculates the nuclei field gradients on coordinates
//args: [start, end] (numpy.ndarray with start and end as entries)
static PyObject* nuclei_field_gradients(PyObject* self, PyObject* args) {
    PyObject *start_end_obj;
    if (!PyArg_ParseTuple(args, "O", &start_end_obj)) {
        return NULL;
    }
    int start = (int)read_vector(start_end_obj)(0);
    int end = (int)read_vector(start_end_obj)(1);
    return (PyObject *)std_vec_of_eigen_matrixXd_to_numpy(computation::nuclei_field_gradients(start, end));
}


Eigen::MatrixXi generateIdxPairs(int start_index, int end_index) {
    int num_atoms = static_cast<int>(global::coordinates.size());
    Eigen::MatrixXi idx_pairs(2, num_atoms * (num_atoms - 1) / 2);
    int k = 0;
    for (int i = 0; i < num_atoms; ++i) {
        for (int j = i + 1; j < num_atoms; ++j) {
            idx_pairs(0, k) = i;
            idx_pairs(1, k) = j;
            k++;
        }
    }
    Eigen::MatrixXi local_idx_pairs(2, end_index - start_index);
    for (int i = start_index; i < end_index; ++i) {
        local_idx_pairs.col(i - start_index) = idx_pairs.col(i);
    }
    return local_idx_pairs;
}


//Calculates self energy of ClassicalSystem for array of indexes
//args: [start, end] (numpy.ndarray with start and end as entries)
static PyObject* self_energy(PyObject* self, PyObject* args) {
    PyObject *start_end_obj;
    if (!PyArg_ParseTuple(args, "O", &start_end_obj)) {
        return NULL;
    }
    int start = (int)read_vector(start_end_obj)(0);
    int end = (int)read_vector(start_end_obj)(1);
    Eigen::MatrixXi idx_arr = generateIdxPairs(start, end);
    return PyFloat_FromDouble(computation::self_energy(idx_arr));
}

//Calculates self energy of ClassicalSystem for array of indexes
//args: [start, end] (numpy.ndarray with start and end as entries)
static PyObject* e_nuc_es(PyObject* self, PyObject* args) {
    PyObject *start_end_obj;
    if (!PyArg_ParseTuple(args, "O", &start_end_obj)) {
        return NULL;
    }
    int start = (int)read_vector(start_end_obj)(0);
    int end = (int)read_vector(start_end_obj)(1);
    return PyFloat_FromDouble(computation::e_nuc_es(start, end));
}

// Sets the global LJ 6-12 parameters sigma and epsilon for a ClassicalSubsystem and QuantumSubsystem.
// args: [classical_sigmas, classical_epsilons, quantum_sigmas, quantum_epsilons];
static PyObject* set_atoms_nuclei_coordinates_lj_sigma_epsilon(PyObject* self, PyObject* args) {
    PyObject *classical_sigmas_obj, *classical_epsilons_obj, *atom_coords_obj, *quantum_sigmas_obj, *quantum_epsilons_obj, *nuc_coords_obj;
    if (!PyArg_ParseTuple(args, "OOOOOO", &classical_sigmas_obj, &classical_epsilons_obj, &atom_coords_obj, &quantum_sigmas_obj, &quantum_epsilons_obj, &nuc_coords_obj)) {
        return NULL;
    }
    global::classical_sigmas = read_vector_d(classical_sigmas_obj);
    global::classical_epsilons = read_vector_d(classical_epsilons_obj);
    global::quantum_sigmas = read_vector_d(quantum_sigmas_obj);
    global::quantum_epsilons = read_vector_d(quantum_epsilons_obj);
        Eigen::MatrixXd coords = read_matrix_d((PyArrayObject *)atom_coords_obj);
    global::coordinates = std::vector<Eigen::Vector3d>();
    for(int i = 0; i < coords.rows(); i++) {
        Eigen::Vector3d coord;
        coord << coords(i, 0), coords(i, 1), coords(i, 2);
        global::coordinates.push_back(coord);
    }
    Eigen::MatrixXd nuc_coords = read_matrix_d((PyArrayObject *)nuc_coords_obj);
    global::nuclei_coordinates = std::vector<Eigen::Vector3d>();
    for(int i = 0; i < nuc_coords.rows(); i++) {
        Eigen::Vector3d coord;
        coord << nuc_coords(i, 0), nuc_coords(i, 1), nuc_coords(i, 2);
        global::nuclei_coordinates.push_back(coord);
    }
    Py_RETURN_NONE;
}

static PyObject* set_combination_rule(PyObject* self, PyObject* args) {
    PyObject *combination_rule_obj;

    if (!PyArg_ParseTuple(args, "O", &combination_rule_obj)) {
        return NULL;
    }
    if (!PyUnicode_Check(combination_rule_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected a string object");
        throw std::invalid_argument("Expected a string object");
    }
    // Get the UTF-8 encoded string data from the Python string
    const char* utf8Str = PyUnicode_AsUTF8(combination_rule_obj);
    if (!utf8Str) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to convert Python string to UTF-8");
        throw std::runtime_error("Failed to convert Python string to UTF-8");
    }
    // Create a C++ std::string from the UTF-8 encoded data
    global::combination_rule = std::string(utf8Str);
    Py_RETURN_NONE;
}


// Computes the VdW potential between a ClassicalSubsystem and a QuantumSubsystem.
//args: [start, end] (numpy.ndarray with start and end as entries)
static PyObject* unperturbed_lj_repulsion(PyObject* self, PyObject* args) {
    PyObject *start_end_obj;
    if (!PyArg_ParseTuple(args, "O", &start_end_obj)) {
        return NULL;
    }
    int start = (int)read_vector(start_end_obj)(0);
    int end = (int)read_vector(start_end_obj)(1);
    return PyFloat_FromDouble(computation::compute_unperturbed_lj_repulsion(start, end, global::combination_rule));
}

// Computes the VdW potential between a ClassicalSubsystem and a QuantumSubsystem.
//args: [start, end] (numpy.ndarray with start and end as entries)
static PyObject* unperturbed_lj_dispersion(PyObject* self, PyObject* args) {
    PyObject *start_end_obj;
    if (!PyArg_ParseTuple(args, "O", &start_end_obj)) {
        return NULL;
    }
    int start = (int)read_vector(start_end_obj)(0);
    int end = (int)read_vector(start_end_obj)(1);
    return PyFloat_FromDouble(computation::compute_unperturbed_lj_dispersion(start, end, global::combination_rule));
}

// Function to convert std::vector<Eigen::Vector3d> to NumPy array
PyObject* std_vec_of_eigen_vec3d_to_numpy(const std::vector<Eigen::Vector3d>& vec) {
    int rows = vec.size();
    int cols = 3; // Eigen::Vector3d has 3 elements

    // Create a NumPy array
    npy_intp dims[2] = {rows, cols};
    PyObject* numpyArray = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

    // Get pointer to data
    double* dataPtr = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(numpyArray)));

    // Copy data from vector to NumPy array
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dataPtr[i * cols + j] = vec[i](j);
        }
    }

    return numpyArray;
}

// Computes the LJ repulsion gradients of the Nuclei in a QuantumSubsystem interacting with a ClassicalSubsystem.
//args: [start, end] (numpy.ndarray with start and end as entries)
static PyObject* lj_repulsion_gradient(PyObject* self, PyObject* args) {
    PyObject *start_end_obj;
    if (!PyArg_ParseTuple(args, "O", &start_end_obj)) {
        return NULL;
    }
    int start = (int)read_vector(start_end_obj)(0);
    int end = (int)read_vector(start_end_obj)(1);
    return std_vec_of_eigen_vec3d_to_numpy(computation::compute_lj_repulsion_gradient(start, end, global::combination_rule));
}

// Computes the LJ dispersion gradients of the Nuclei in a QuantumSubsystem interacting with a ClassicalSubsystem.
//args: [start, end] (numpy.ndarray with start and end as entries)
static PyObject* lj_dispersion_gradient(PyObject* self, PyObject* args) {
    PyObject *start_end_obj;
    if (!PyArg_ParseTuple(args, "O", &start_end_obj)) {
        return NULL;
    }
    int start = (int)read_vector(start_end_obj)(0);
    int end = (int)read_vector(start_end_obj)(1);
    return std_vec_of_eigen_vec3d_to_numpy(computation::compute_lj_dispersion_gradient(start, end, global::combination_rule));
}


// Method table for the module
static PyMethodDef module_methods[] = {
    {"compute_interaction_tensor_element", compute_interaction_tensor_element, METH_VARARGS,
     "Computes Interaction Tensor Element."},
    {"compute_t_tensor", compute_t_tensor, METH_VARARGS,
     "Computes t_tensor."},
    {"set_tensor_coefficients", set_tensor_coefficients, METH_VARARGS,
     "Sets tensor coefficients and templates for interaction and potential tensors."},
    {"set_coords_idxs_exlcs", set_coords_idxs_exlcs, METH_VARARGS,
     "Sets coordinates, indices and exclusions for the inner loop of the solver."},
    {"set_old_ind_dipoles", set_old_ind_dipoles, METH_VARARGS,
     "Sets the old induced dipole fields for the calculation of the induced dipoles."},
    {"ind_dipoles_fields", ind_dipoles_fields, METH_VARARGS,
     "Calculates induced dipoles fields at atom i from old induced dipoles and previously set coords, idxs and exclusions."},
     {"set_coords_nuc_coords_charges", set_coords_nuc_coords_charges, METH_VARARGS,
     "Sets atom coordinates, nuclear coordinates, and nuclear charges for the calculation of nuclei fields."},
     {"nuclei_fields", nuclei_fields, METH_VARARGS,
     "Calculates the field of the nuclei on atoms defined with start and end. Previously set coordinates, nuclei_coords and nuclei_charges."},
     {"nuclei_field_gradients", nuclei_field_gradients, METH_VARARGS,
      "Calculates the field gradients of the nuclei on atoms defined with start and end. Previously set coordinates, nuclei_coords and nuclei_charges."},
     {"set_multipoles_multipoles_order", set_multipoles_multipoles_order, METH_VARARGS,
     "Sets multipoles with degeneracy and taylor coefficient and the multipole orders."},
     {"multipole_fields", multipole_fields, METH_VARARGS,
     "Calculates the field of the multipoles at atom i. Previously set coords, idxs, exclusions, multipoles, multipole_orders."},
     {"self_energy", self_energy, METH_VARARGS,
     "Calculates the self energy of a ClassicalSubsystem. Previously set coords, idxs, exclusions, multipoles, multipole_orders."},
     {"e_nuc_es", e_nuc_es, METH_VARARGS,
     "Calculates the electrostatic energy between all Atoms and Nuclei. Previously set coords, multipoles, multipole_orders, nuclei_coords and nuclei_charges."},
     {"set_atoms_nuclei_coordinates_lj_sigma_epsilon", set_atoms_nuclei_coordinates_lj_sigma_epsilon, METH_VARARGS,
     "Sets the LJ 6-12 parameters of a ClassicalSubsystem and a QuantumSubsystem."},
     {"set_combination_rule", set_combination_rule, METH_VARARGS,
     "Sets the combination rule for non-bonded VdW interactions."},
     {"unperturbed_lj_repulsion", unperturbed_lj_repulsion, METH_VARARGS,
     "Computes the LJ repulsion between a ClassicalSubsystem and a QuantumSubsystem."},
     {"unperturbed_lj_dispersion", unperturbed_lj_dispersion, METH_VARARGS,
     "Computes the LJ repulsion between a ClassicalSubsystem and a QuantumSubsystem."},
     {"lj_repulsion_gradient", lj_repulsion_gradient, METH_VARARGS,
     "Computes the LJ repulsion potential between a ClassicalSubsystem and a QuantumSubsystem."},
     {"lj_dispersion_gradient", lj_dispersion_gradient, METH_VARARGS,
     "Computes the LJ dispersion potential between a ClassicalSubsystem and a QuantumSubsystem."},
    {NULL, NULL, 0, NULL}};

// Module definition
static struct PyModuleDef engine = {
    PyModuleDef_HEAD_INIT,
    "engine", // Module name
    NULL,                     // Module documentation
    -1,                       // Size of per-interpreter state
    module_methods};

// Module initialization
PyMODINIT_FUNC PyInit_engine(void)
{
    import_array();
    Py_Initialize();
    assert(!PyErr_Occurred());
    if (PyErr_Occurred()) {
        std::cerr << "Failed to import numpy Python module(s)." << std::endl;
        return NULL;
    }
    return PyModule_Create(&engine);
}
