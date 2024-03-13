// Some core functionality implemented in C++
// Multiprocessing using OpenMP is partially implemented.
// Caution is advised regarding the validity of inputs, as some common errors are checked, but not all.
// Undefined states may result from invalid inputs.

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "numpy/arrayobject.h"

#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <unordered_set>

#include "python_eigen_conversion.h"
#include "computation.h"
#include "global.h"

// Computes an interaction tensor element from a given multiindex, r_ab == r_b - r_a, and tensor_coefficients
// args: [multiindex, r_ab, tensor_coefficients]
static PyObject* compute_interaction_tensor_element(PyObject* self, PyObject* args) {
    PyObject *multiindex_obj, *r_ab_array;

    if (!PyArg_ParseTuple(args, "OO", &multiindex_obj, &r_ab_array))
    {
        return NULL;
    }

    Eigen::Matrix<int, 2, 3> multiindex = conversion::read_multiindex(multiindex_obj);
    Eigen::Vector3d r_ab = conversion::read_vector3d((PyArrayObject *)r_ab_array);

    double result = computation::compute_interaction_tensor_element(multiindex, r_ab, global::tensor_coefficients);

    return Py_BuildValue("d", result);
}


// Computes the t_tensor for the interaction of two atoms in a specified range of indices.
// args: [r_a, r_b, rank_a, rank_b, start_rank_a, start_rank_b, is_potential (true: potential, false: interaction)]
static PyObject* compute_t_tensor(PyObject* self, PyObject* args) {
    PyObject *r_a_obj, *r_b_obj;
    long long rank_a, rank_b, start_rank_a, start_rank_b;
    bool is_potential;

    if (!PyArg_ParseTuple(args, "OOLLLLp", &r_a_obj, &r_b_obj,
                            &rank_a, &rank_b, &start_rank_a, &start_rank_b, &is_potential)) {
        return NULL;
    }

    Eigen::Vector3d r_a = conversion::read_vector3d((PyArrayObject *)r_a_obj);
    Eigen::Vector3d r_b = conversion::read_vector3d((PyArrayObject *)r_b_obj);
    if(r_a[0] == r_b[0] && r_a[1] == r_b[1] && r_a[2] == r_b[2])
    {
        PyErr_SetString(PyExc_ValueError, "r_a and r_b cannot be equal.");
        return conversion::eigen_matrix_to_numpy(Eigen::MatrixXd(0, 0));
    }
    // TODO: Implement a mechanism ensuring sufficient array size
    // or make sure the first call to this function has the maximum required size (which seems to be the case?)

    Eigen::MatrixXd t_tensor = computation::compute_t_tensor(r_b - r_a,
                                is_potential ? global::tensor_template_potential : global::tensor_template_interaction,
                                (int)rank_a, (int)rank_b, (int)start_rank_a, (int)start_rank_b);

    return conversion::eigen_matrix_to_numpy(t_tensor);
}

// Sets the global tensor coefficients and templates
// args: [tensor_coefficients, tensor_template_interaction, tensor_template_potential, rank, max_order]
static PyObject* set_tensor_coefficients(PyObject* self, PyObject* args) {
    PyObject *tensor_coefficients_obj, *tensor_template_interaction_obj, *tensor_template_potential_obj;
    long long rank, max_order;
    if (!PyArg_ParseTuple(args, "OOOLL", &tensor_coefficients_obj, &tensor_template_interaction_obj,
                          &tensor_template_potential_obj, &rank, &max_order)) {
        return NULL;
    }
    if(rank > global::rank) {
        // Check if the input is a valid 2D NumPy array
        if (!PyArray_Check(tensor_template_interaction_obj) || PyArray_NDIM((PyArrayObject*)tensor_template_interaction_obj) != 2) {
            PyErr_SetString(PyExc_TypeError, "tensor_template must be a 2D NumPy array");
            Py_RETURN_NONE;
        }
        global::tensor_template_interaction = conversion::read_tensor_template((PyArrayObject *)tensor_template_interaction_obj);
        global::tensor_template_potential = conversion::read_tensor_template((PyArrayObject *)tensor_template_potential_obj);
        global::rank = (long)rank;
    }
    if(max_order > global::max_order) {
        global::tensor_coefficients = conversion::read_tensor((PyArrayObject *)tensor_coefficients_obj);
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
    Eigen::MatrixXd coords = conversion::read_matrix((PyArrayObject *)coords_obj);
    global::coordinates = std::vector<Eigen::Vector3d>();
    for(int i = 0; i < coords.rows(); i++) {
        Eigen::Vector3d coord;
        coord << coords(i, 0), coords(i, 1), coords(i, 2);
        global::coordinates.push_back(coord);
    }
    global::indices = conversion::read_vector((PyArrayObject *)indices_obj);

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

//
static PyObject* ind_dipoles_fields(PyObject* self, PyObject* args) {
    PyObject *old_ind_dipoles_obj;
    long long i;
    if (!PyArg_ParseTuple(args, "Oi", &old_ind_dipoles_obj, &i)) {
        return NULL;
    }
    Eigen::MatrixXd old_ind_dipoles = conversion::read_matrix((PyArrayObject *)old_ind_dipoles_obj);
    return (PyObject *)conversion::eigen_matrix_to_numpy(computation::ind_dipoles_field(old_ind_dipoles, (int)i));
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
    {"ind_dipoles_fields", ind_dipoles_fields, METH_VARARGS,
     "Calculates induced dipoles fields at atom i from old induced dipoles and previously set coords, idxs and exclusions"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef engine = {
    PyModuleDef_HEAD_INIT,
    "engine", // Module name
    NULL,                     // Module documentation
    -1,                       // Size of per-interpreter state
    module_methods};

// Module initialization
PyMODINIT_FUNC PyInit_engine(void) {
    assert(!PyErr_Occurred());
    import_array();
    conversion::init();
    if (PyErr_Occurred()) {
        std::cerr << "Failed to import numpy Python module(s)." << std::endl;
        return NULL;
    }
    return PyModule_Create(&engine);
}
