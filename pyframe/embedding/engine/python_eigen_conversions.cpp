#include "python_eigen_conversions.h"

namespace conversion
{

//Reads a multiindex into an Eigen::Matrix.
//multiindex_obj: Points to a List of two np.ndarrays of length 3
Eigen::Matrix<int, 2, 3> read_multiindex(PyObject* multiindex_obj) {
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
Eigen::Vector3d read_vector3d(PyObject* array_obj) {
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
std::vector<Eigen::MatrixXd> read_tensor(PyObject* tensor_obj) {
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
    return tensor;
}

//Reads a 2D np.ndarray into an Eigen::MatrixXd.
//matrix_obj: Points to a 3-dimensional np.ndarray contining tensor coefficients
Eigen::MatrixXd read_matrix(PyObject* matrix_obj) {
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
Eigen::VectorXi read_vector(PyObject* vector_obj) {
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
PyObject* eigen_matrix_to_numpy(Eigen::MatrixXd &matrix) {
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
Eigen::Matrix<Eigen::Matrix<int, 2, 3>, Eigen::Dynamic, Eigen::Dynamic> read_tensor_template(PyObject* template_obj) {
    // Check if the input is a valid 2D NumPy array
    if (!PyArray_Check(template_obj) || PyArray_NDIM(template_obj) != 2) {
        PyErr_SetString(PyExc_TypeError, "tensor_template must be a 2D NumPy array");
        return Eigen::Matrix<Eigen::Matrix<int, 2, 3>, Eigen::Dynamic, Eigen::Dynamic>(0, 0);
    }
    npy_intp* dims = PyArray_DIMS(template_obj);
    npy_intp rows = dims[0];
    npy_intp cols = dims[1];

    Eigen::Matrix<Eigen::Matrix<int, 2, 3>, Eigen::Dynamic, Eigen::Dynamic> tensor_template(rows, cols);

    for (npy_intp i = 0; i < rows; ++i) {
        for (npy_intp j = 0; j < cols; ++j) {
            PyObject* element = PyArray_GETITEM(template_obj, PyArray_GETPTR2(template_obj, i, j));
            tensor_template(i,j) = conversion::read_multiindex(element);
        }
    }
    return tensor_template;
}

//Must be called at least once before using any other function.
void init() {
    _import_array();
}
}

