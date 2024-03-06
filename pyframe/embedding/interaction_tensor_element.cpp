#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/arrayobject.h" // Include any other Numpy headers, UFuncs for example.
#include <stdlib.h>
#include <iostream>
#include <cmath>

// C++ Version of compute_interaction_tensor_element from tensor_tools.py

extern "C" {

    static PyObject* compute_interaction_tensor_element(PyObject* self, PyObject* args) {
        // Parse the input arguments
        PyObject* multiindex_obj;
        PyObject* dist_obj;
        PyObject* tensor_obj;

        if (!PyArg_ParseTuple(args, "OOO", &multiindex_obj, &dist_obj, &tensor_obj)) {
            return NULL;
        }

        // Check if multiindex is a list
        if (!PyList_Check(multiindex_obj)) {
            PyErr_SetString(PyExc_TypeError, "multiindex must be a list of NumPy arrays");
            return NULL;
        }

        // Check if dist and tensor are NumPy arrays
        if (!PyArray_Check(dist_obj) || !PyArray_Check(tensor_obj)) {
            PyErr_SetString(PyExc_TypeError, "dist and tensor must be NumPy arrays");
            return NULL;
        }

		//Extract multiindex data
        int ax, ay, az, bx, by, bz;

        // Extract data from the first NumPy array in multiindex, holding values ax, ay, az
        PyObject* array1 = PyList_GetItem(multiindex_obj, 0);
        if (!PyArray_Check(array1) || PyArray_SIZE((PyArrayObject*)array1) != 3) {
            PyErr_SetString(PyExc_ValueError, "Each array in multiindex must be a NumPy array of length 3");
            return NULL;
        }
        
        PyArray_Descr *descr1 = PyArray_DESCR((PyArrayObject *)array1);
        if (descr1 == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Failed to get array descriptor");
            return NULL;
        }        
        int type_num1 = descr1->type_num;

        if (type_num1 == NPY_INT32) {

            int32_t *multiindex_data1 = static_cast<int32_t *>(PyArray_DATA((PyArrayObject *)array1));
            ax = multiindex_data1[0];
            ay = multiindex_data1[1];
            az = multiindex_data1[2];
        }
        else if (type_num1 == NPY_INT64) {
            int64_t *multiindex_data1 = static_cast<int64_t *>(PyArray_DATA((PyArrayObject *)array1));
            ax = multiindex_data1[0];
            ay = multiindex_data1[1];
            az = multiindex_data1[2];
        }
        else {
            PyErr_SetString(PyExc_TypeError, "Unsupported data type");
            return NULL;
        }


        // Extract data from the second NumPy array in multiindex, hloding values bx, by, bz
        PyObject* array2 = PyList_GetItem(multiindex_obj, 1);
        if (!PyArray_Check(array2) || PyArray_SIZE((PyArrayObject*)array2) != 3) {
            PyErr_SetString(PyExc_ValueError, "Each array in multiindex must be a NumPy array of length 3");
            return NULL;
        }
        
        PyArray_Descr *descr2 = PyArray_DESCR((PyArrayObject *)array2);
        if (descr2 == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to get array descriptor");
            return NULL;
        }
        
        int type_num2 = descr2->type_num;

        // Process the array based on the data type
        if (type_num2 == NPY_INT32) {

            int32_t *multiindex_data2 = static_cast<int32_t *>(PyArray_DATA((PyArrayObject *)array2));
            bx = multiindex_data2[0];
            by = multiindex_data2[1];
            bz = multiindex_data2[2];
        }
        else if (type_num2 == NPY_INT64) {
            int64_t *multiindex_data2 = static_cast<int64_t *>(PyArray_DATA((PyArrayObject *)array2));
            bx = multiindex_data2[0];
            by = multiindex_data2[1];
            bz = multiindex_data2[2];
        }
        else {
            PyErr_SetString(PyExc_TypeError, "Unsupported data type");
            return NULL;
        }

        double* dist_data = static_cast<double*>(PyArray_DATA((PyArrayObject*)dist_obj));
        if (!PyArray_Check(array2) || PyArray_SIZE((PyArrayObject*)array2) != 3) {
            PyErr_SetString(PyExc_ValueError, "distance_vector must be a NumPy array of length 3");
            return NULL;
        }
        double dx = dist_data[0], dy = dist_data[1], dz = dist_data[2];

        int i = ax + bx, j = ay + by, k = az + bz;
        double norm = std::sqrt(dx * dx + dy * dy + dz * dz);

        if (!PyArray_Check(tensor_obj) || PyArray_NDIM((PyArrayObject*)tensor_obj) != 3) {
            PyErr_SetString(PyExc_ValueError, "b must be a three-dimensional NumPy array");
            return NULL;
        }

        // Get max dimensions of array
        int t = PyArray_DIM((PyArrayObject*)tensor_obj, 0);
        int u = PyArray_DIM((PyArrayObject*)tensor_obj, 1);
        int v = PyArray_DIM((PyArrayObject*)tensor_obj, 2);

        // Check if indices are within bounds
        int min_size_xy = std::max(std::max(i, j), k) + 1;
        int min_size_z = 2 * i + 2 * j + 1;
        if (t < min_size_xy || u < min_size_xy || v < min_size_z) {
            PyErr_SetString(PyExc_IndexError, "tensor dimensions too small");
            return NULL;
        }

        // Access the element at indices i, j, k
        double* tensor_data = static_cast<double*>(PyArray_DATA((PyArrayObject*)tensor_obj));
        double element = 0;
        auto getTensorCoeff = [=](int idx0, int idx1, int idx2) {return tensor_data[idx0 * u * v + idx1 * v + idx2];};

        for (int q = 0; q <= i; q++) {
            double cl = getTensorCoeff(q, i, 1) * std::pow(dx/norm, q);
            int o = q + i + 1;
            for (int m = 0; m <= j; m++) {
                double cm = cl * getTensorCoeff(m, j, o) * std::pow(dy / norm, m);
                int p = o + j + m;
                for (int n = 0; n <= k; n++) {
                    double cn = cm * getTensorCoeff(n, k, p) * std::pow(dz / norm, n);
                    element += cn;
                }
            }
        }
        element /= std::pow(norm, i + j + k + 1);
        return PyFloat_FromDouble(element * std::pow(-1, bx + by + bz));
    }

    // Method table for the module
    static PyMethodDef module_methods[] = {
        {"compute_interaction_tensor_element", compute_interaction_tensor_element, METH_VARARGS,
         "Computes Interaction Tensor Element."},
        {NULL, NULL, 0, NULL}  // Sentinel
    };

    // Module definition
    static struct PyModuleDef cpp_interaction_tensor_element = {
        PyModuleDef_HEAD_INIT,
        "cpp_interaction_tensor_element",   // Module name
        NULL,         // Module documentation
        -1,           // Size of per-interpreter state
        module_methods
    };

    // Module initialization
    PyMODINIT_FUNC PyInit_cpp_interaction_tensor_element(void) {
        assert(!PyErr_Occurred());
        // Initialise Numpy
        import_array();
        if (PyErr_Occurred()) {
            std::cerr << "Failed to import numpy Python module(s)." << std::endl;
            return NULL; // Or some suitable return value to indicate failure.
        }
        return PyModule_Create(&cpp_interaction_tensor_element);
    }
}
