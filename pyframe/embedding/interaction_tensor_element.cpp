#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/arrayobject.h" // Include any other Numpy headers, UFuncs for example.
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <unordered_set>
#include <omp.h>

// C++ Version of compute_interaction_tensor_element and compute_t_tensor from tensor_tools.py
// Caution is advised regarding the validity of inputs, as some common errors are checked, but not all.
// Undefined states may result from invalid inputs.

// TODO: ensure error handling tests are passed when r_a == r_b
// TODO: Improve performance by parsing tensor_template and tensor_coefficients only once. See lines 344 and following.

int rank_global = -1;
int max_order_global = -1;
std::vector<Eigen::MatrixXd> tensor_coefficients_global;
Eigen::Matrix<Eigen::Matrix<int, 2, 3>, Eigen::Dynamic, Eigen::Dynamic> tensor_template_interaction_global;
Eigen::Matrix<Eigen::Matrix<int, 2, 3>, Eigen::Dynamic, Eigen::Dynamic> tensor_template_potential_global;

//Reads a multiindex into an Eigen::Matrix.
//multiindex_obj: Points to a List of two np.ndarrays of length 3
static Eigen::Matrix<int, 2, 3> read_multiindex(PyObject* multiindex_obj)
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

//Reads a vector of length 3 into an Eigen::Vector3d.
//array_obj: Points to an np.ndarray of length 3
static Eigen::Vector3d read_vector3d(PyObject* array_obj)
{
    if (!PyArray_Check(array_obj))
    {
        PyErr_SetString(PyExc_TypeError, "vector must be NumPy array");
        return Eigen::Vector3d();
    }
    if (PyArray_NDIM(array_obj) != 1) {
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

//Reads a 3D tensor into a std..vector<Eigen::MatrixXd>. tensor[i, j, k] == return_value[i](j, k).
//tensor_obj: Points to a 3-dimensional np.ndarray contining tensor coefficients
static std::vector<Eigen::MatrixXd> read_tensor(PyObject* tensor_obj)
{
    // Check if tensor is NumPy array
    if (!PyArray_Check(tensor_obj))
    {
        PyErr_SetString(PyExc_TypeError, "tensor must be NumPy array");
        return std::vector<Eigen::MatrixXd>();
    }
    if (PyArray_NDIM(tensor_obj) != 3) {
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

    //TODO: Check sufficiency of size of tensor
    return tensor;
}

//Reads a 2D np.ndarray into an Eigen::MatrixXd.
//matrix_obj: Points to a 3-dimensional np.ndarray contining tensor coefficients
static Eigen::MatrixXd read_matrix(PyObject* matrix_obj)
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

//Reads a 1D np.ndarray into an Eigen::VectorXd.
//matrix_obj: Points to a 3-dimensional np.ndarray contining tensor coefficients
static Eigen::VectorXi read_vector(PyObject* vector_obj)
{
    if (!PyArray_Check(vector_obj) || PyArray_NDIM(vector_obj) != 1)
    {
        PyErr_SetString(PyExc_TypeError, "vector must be a one-dimensional NumPy array");
        return Eigen::VectorXi();
    }
    npy_intp u = PyArray_DIM(vector_obj, 0);
    Eigen::VectorXi vector(u);
    for (int j = 0; j < u; ++j)
    {
        vector(j) = (int) *(long *)PyArray_GETPTR1((PyArrayObject *)vector_obj, j);
    }
    return vector;
}

//Creates, from an Eigen::MatrixXd, a 2-dimensional np.ndarray of the same shape.
//matrix: the matrix to use
static PyObject* eigen_matrix_to_numpy(Eigen::MatrixXd &matrix)
{
    int rows = (int)matrix.rows();
    int cols = (int)matrix.cols();

    npy_intp dims[] = {rows, cols};
    PyObject* numpyArray = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (numpyArray == NULL) {
        return NULL;  // Memory allocation failed
    }

    // Copy data from Eigen matrix to NumPy array, but first converting to row-major order
    double* numpyData = static_cast<double*>(PyArray_DATA(numpyArray));
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_rm = matrix;
    const double* eigenData = matrix_rm.data();
    std::copy(eigenData, eigenData + rows * cols, numpyData);

    return numpyArray;
}

//Reads the tensor template. template[i, j][k][l] = return_value(i, j)(k, l).
//template_obj: 2-dimensional np.ndarray of lists (length 2) of 1-dimensional np.ndarrays (length 3).
static Eigen::Matrix<Eigen::Matrix<int, 2, 3>, Eigen::Dynamic, Eigen::Dynamic> read_tensor_template(PyObject* template_obj)
{
    // Check if the input is a valid 2D NumPy array
    if (!PyArray_Check(template_obj) || PyArray_NDIM(template_obj) != 2) {
        PyErr_SetString(PyExc_TypeError, "tensor_template must be a 2D NumPy array");
        return Eigen::Matrix<Eigen::Matrix<int, 2, 3>, Eigen::Dynamic, Eigen::Dynamic>(0, 0);
    }

    npy_intp* dims = PyArray_DIMS(template_obj);
    npy_intp rows = dims[0];
    npy_intp cols = dims[1];

    Eigen::Matrix<Eigen::Matrix<int, 2, 3>, Eigen::Dynamic, Eigen::Dynamic> tensor_template(rows, cols);

    for (npy_intp i = 0; i < rows; ++i)
    {
        for (npy_intp j = 0; j < cols; ++j)
        {
            PyObject* element = PyArray_GETITEM(template_obj, PyArray_GETPTR2(template_obj, i, j));
            tensor_template(i,j) = read_multiindex(element);
        }
    }
    return tensor_template;
}

//Calculates the length of a polytensor given start_rank and end_rank in that dimension.
static int get_polytensor_length(int start_rank, int end_rank)
{
    int length = 0;
    for(int i = start_rank; i <= end_rank; i++) {
        length += (i + 1) * (i + 2) / 2;
    }
    return length;
}

//Computes an interaction tensor element from a given multiindex, r_ab == r_b - r_a, and tensor_coefficients
static double compute_interaction_tensor_element(
    const Eigen::Matrix<int, 2, 3> &multiindex,
    const Eigen::Vector3d &r_ab,
    const std::vector<Eigen::MatrixXd> &tensor_coefficients)
{
    int i = multiindex.col(0).sum(), j = multiindex.col(1).sum(), k = multiindex.col(2).sum();
    double element = 0;

    // TODO: divide r_ab by norm before the loops to optimize performance
    double norm = r_ab.norm();

    for (int q = 0; q <= i; q++)
    {
        double cl = tensor_coefficients[q](i, 1) * std::pow(r_ab(0) / norm, q);
        int o = q + i + 1;
        for (int m = 0; m <= j; m++)
        {
            double cm = cl * tensor_coefficients[m](j, o) * std::pow(r_ab(1) / norm, m);
            int p = o + j + m;
            for (int n = 0; n <= k; n++)
            {
                double cn = cm * tensor_coefficients[n](k, p) * std::pow(r_ab(2) / norm, n);
                element += cn;
            }
        }
    }
    element /= std::pow(norm, i + j + k + 1);
    element *= std::pow(-1, multiindex.row(1).sum());

    return element;
}

//Computes the t_tensor for the interaction of two atoms in a specified range of indices.
static Eigen::MatrixXd compute_t_tensor(
    const Eigen::Vector3d &r_ab,
    const Eigen::Matrix<Eigen::Matrix<int, 2, 3>, Eigen::Dynamic, Eigen::Dynamic> &tensor_template,
    const std::vector<Eigen::MatrixXd> &tensor_coefficients,
    int rank_a,
    int rank_b,
    int start_rank_a,
    int start_rank_b)
{
    if(r_ab[0] == 0 && r_ab[1] == 0 && r_ab[2] == 0)
    {
        PyErr_SetString(PyExc_ValueError, "r_a and r_b cannot be equal.");
        return Eigen::MatrixXd(0, 0);
    }
    Eigen::MatrixXd interaction_tensor(get_polytensor_length(start_rank_a, rank_a),
                                        get_polytensor_length(start_rank_b, rank_b));

    int start_b = (start_rank_b) * (start_rank_b + 1) * (start_rank_b + 2) / 6;
    int end_b = (rank_b + 1) * (rank_b + 2) * (rank_b + 3) / 6;
    int start_a = (start_rank_a) * (start_rank_a + 1) * (start_rank_a + 2) / 6;
    int end_a = (rank_a + 1) * (rank_a + 2) * (rank_a + 3) / 6;

    for(int i = start_a; i < end_a; i++)
    {
        for(int j = start_b; j < end_b; j++)
        {
            double interaction_element = compute_interaction_tensor_element(
                tensor_template(i, j), r_ab, tensor_coefficients);
            interaction_tensor(i - start_a, j - start_b) = interaction_element;
        }
    }

    return interaction_tensor;
}

//Computes an interaction tensor element from a given multiindex, r_ab == r_b - r_a, and tensor_coefficients
//args: [multiindex, r_ab, tensor_coefficients]
static PyObject* compute_interaction_tensor_element_py(PyObject* self, PyObject* args)
{
    PyObject *multiindex_obj, *r_ab_array;

    if (!PyArg_ParseTuple(args, "OO", &multiindex_obj, &r_ab_array))
    {
        return NULL;
    }

    Eigen::Matrix<int, 2, 3> multiindex = read_multiindex(multiindex_obj);
    Eigen::Vector3d r_ab = read_vector3d(r_ab_array);

    double result = compute_interaction_tensor_element(multiindex, r_ab, tensor_coefficients_global);

    return Py_BuildValue("d", result);
}


//Computes the t_tensor for the interaction of two atoms in a specified range of indices.
//args: [r_a, r_b, rank_a, rank_b, start_rank_a, start_rank_b, is_potential (true: potential, false: interaction)]
static PyObject* compute_t_tensor_py(PyObject* self, PyObject* args)
{
    PyObject *r_a_obj, *r_b_obj;
    int rank_a, rank_b, start_rank_a, start_rank_b;
    bool is_potential;

    if (!PyArg_ParseTuple(args, "OOiiiip", &r_a_obj, &r_b_obj,
                            &rank_a, &rank_b, &start_rank_a, &start_rank_b, &is_potential)) {
        return NULL;
    }

    Eigen::Vector3d r_a = read_vector3d(r_a_obj);
    Eigen::Vector3d r_b = read_vector3d(r_b_obj);

    // TODO: Implement a mechanism ensuring sufficient array size
    // or make sure the first call to this function has the maximum required size (which seems to be the case?)

    Eigen::MatrixXd t_tensor = compute_t_tensor(r_b - r_a,
                                is_potential ? tensor_template_potential_global : tensor_template_interaction_global,
                                tensor_coefficients_global, rank_a, rank_b, start_rank_a, start_rank_b);

    return eigen_matrix_to_numpy(t_tensor);
}

static PyObject* set_tensor_coefficients(PyObject* self, PyObject* args)
{
    PyObject *tensor_coefficients_obj, *tensor_template_interaction_obj, *tensor_template_potential_obj;
    int rank, max_order;
    if (!PyArg_ParseTuple(args, "OOOii", &tensor_coefficients_obj, &tensor_template_interaction_obj,
                            &tensor_template_potential_obj, &rank, &max_order)) {
        return NULL;
    }
    if(rank > rank_global) {
        tensor_template_interaction_global = read_tensor_template(tensor_template_interaction_obj);
        tensor_template_potential_global = read_tensor_template(tensor_template_potential_obj);
        rank_global = rank;
    }
    if(max_order > max_order_global) {
        tensor_coefficients_global = read_tensor(tensor_coefficients_obj);
        max_order_global = max_order;
    }
    Py_RETURN_NONE;
}

std::vector<Eigen::Vector3d> coordinates_global;
Eigen::VectorXi indices_global;
std::vector<std::unordered_set<int>> exclusions_global;

static PyObject* set_coords_idxs_exlcs(PyObject* self, PyObject* args)
{
    PyObject *coords_obj, *indices_obj, *exclusions_obj;

    if (!PyArg_ParseTuple(args, "OOO", &coords_obj, &indices_obj, &exclusions_obj)) {
        return NULL;
    }
    Eigen::MatrixXd coords = read_matrix(coords_obj);
    coordinates_global = std::vector<Eigen::Vector3d>();
    for(int i = 0; i < coords.rows(); i++) {
        Eigen::Vector3d coord;
        coord << coords(i, 0), coords(i, 1), coords(i, 2);
        coordinates_global.push_back(coord);
    }
    indices_global = read_vector(indices_obj);

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
    exclusions_global = exclusions;
    Py_RETURN_NONE;
}

static PyObject* ind_dipoles_fields_py(PyObject* self, PyObject* args)
{
    // TODO maybe not necessary to init parallel? https://eigen.tuxfamily.org/dox/TopicMultiThreading.html
    Eigen::initParallel();
    PyObject *old_ind_dipoles_obj;
    int i;

    if (!PyArg_ParseTuple(args, "Oi", &old_ind_dipoles_obj, &i)) {
        return NULL;
    }
    Eigen::MatrixXd old_ind_dipoles = read_matrix(old_ind_dipoles_obj);
    Eigen::MatrixXd ind_dipoles_fields = Eigen::MatrixXd::Zero(3, 1);

    #pragma omp parallel
    {
        Eigen::MatrixXd my_part = Eigen::MatrixXd::Zero(3, 1);
        #pragma omp for
        for(int j = 0; j < coordinates_global.size(); j++) {
            if(exclusions_global[i].find(indices_global[j]) != exclusions_global[i].end()) {
                continue;
            }
            Eigen::Vector3d r_ab = coordinates_global[i] - coordinates_global[j];
            Eigen::MatrixXd t_tensor = compute_t_tensor(r_ab,
                                                        tensor_template_potential_global,
                                                        tensor_coefficients_global,
                                                        1, 1, 1, 1);
            my_part += t_tensor * old_ind_dipoles.row(j).transpose();
        }
        #pragma omp critical
        {
            ind_dipoles_fields += my_part;
        }
    }

    return eigen_matrix_to_numpy(ind_dipoles_fields);
}

// Method table for the module
static PyMethodDef module_methods[] = {
    {"compute_interaction_tensor_element", compute_interaction_tensor_element_py, METH_VARARGS,
     "Computes Interaction Tensor Element."},
    {"compute_t_tensor", compute_t_tensor_py, METH_VARARGS,
     "Computes t_tensor."},
    {"set_tensor_coefficients", set_tensor_coefficients, METH_VARARGS,
     "Sets tensor coefficients and templates for interaction and potential tensors."},
    {"set_coords_idxs_exlcs", set_coords_idxs_exlcs, METH_VARARGS,
     "Sets coordinates, indices and exclusions for the inner loop of the solver."},
    {"ind_dipoles_fields", ind_dipoles_fields_py, METH_VARARGS,
     "Calculates induced dipoles fields at atom i from old induced dipoles and previously set coords, idxs and exclusions"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef cpp_interaction_tensor_element = {
    PyModuleDef_HEAD_INIT,
    "cpp_interaction_tensor_element", // Module name
    NULL,                     // Module documentation
    -1,                       // Size of per-interpreter state
    module_methods};

// Module initialization
PyMODINIT_FUNC PyInit_cpp_interaction_tensor_element(void)
{
    assert(!PyErr_Occurred());
    import_array();
    if (PyErr_Occurred()) {
        std::cerr << "Failed to import numpy Python module(s)." << std::endl;
        return NULL;
    }
    return PyModule_Create(&cpp_interaction_tensor_element);
}
