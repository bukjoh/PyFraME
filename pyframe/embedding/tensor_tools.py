import numpy as np
from typing import List, Tuple


def compute_factorials(max_order: int) -> np.ndarray:
    """Compute factorials up to a given order.

    Args:
        max_order: The order up to which the factorials will be computed.

    Returns:
        Contains the factorials up to the given order. The factorials are
        stored in ascending order (i! = factorials[i]).
    """
    factorials = np.zeros(max_order + 1)
    factorials[0] = 1
    for i in range(1, max_order + 1):
        factorials[i] = factorials[i - 1] * i
    return factorials


def compute_double_factorials(max_order: int) -> np.ndarray:
    """Compute double factorials up to a given order.

     Args:
         max_order: The order up to which the double factorials are computed.

    Returns:
         Contains the double factorials up to the given order. The double factorials are stored in ascending order
          (i!! = double_factorials[i]).
     """
    double_factorials = np.zeros(max_order + 2)
    double_factorials[0] = 1
    double_factorials[1] = 1
    double_factorials[-1] = 1
    for i in range(2, max_order + 1):
        double_factorials[i] = double_factorials[i - 2] * i
    return double_factorials


def compute_binomial_coefficients(n: int, k: int) -> np.ndarray:
    """Compute binomial coefficients.

    Args:
        n: Number of rows of Pascal's triangle.
        k: Number of columns of Pascal's triangle.

    Returns:
        Array of shape (n + 1, k + 1), containing the binomial coefficients (n,k) which can be accessed through the
        array binomial_coefficients[n, k].
    """
    binomial_coefficients = np.zeros((n + 1, k + 1))
    binomial_coefficients[0, 0] = 1.
    for i in range(1, n + 1):
        binomial_coefficients[i, 0] = 1.
        for j in range(1, min(i + 1, k + 1)):
            binomial_coefficients[i, j] = binomial_coefficients[i - 1, j - 1] + binomial_coefficients[i - 1, j]
    return binomial_coefficients


def compute_trinomial_coefficients(i: int, j: int, k: int, binomial_coefficients: np.ndarray) -> np.ndarray:
    """Compute trinomial coefficients.

    Args:
        i: First index of the trinomial expansion to which the trinomial coefficients are computed.
        j: Second index of the trinomial expansion to which the trinomial coefficients are computed.
        k: Third index of the trinomial expansion to which the trinomial coefficients are computed.
        binomial_coefficients: Array of shape (n + 1, k + 1) containing binomial coefficients, where n needs to be at
        least larger than or equal to (i + j + k) and the k of the binomial coefficients needs to be at least larger
        than or equal to the k used for the trinomial coefficients.

    Returns:
        Array of shape (i + 1, j + 1, k + 1) that contains the trinomial coefficients (n, i j k) which can be accessed
        through the array trinomial_coefficients[i, j, k].
    """
    trinomial_coefficients = np.zeros((i + 1, j + 1, k + 1))
    for o in range(i + 1):
        for p in range(j + 1):
            for q in range(k + 1):
                trinomial_coefficients[o, p, q] = binomial_coefficients[o + p, p] * binomial_coefficients[o + p + q, q]
    return trinomial_coefficients


def compute_degeneracy_tensor(tensor_rank: int, trinomial_coefficients: np.ndarray) -> np.ndarray:
    """Compute the degeneracy tensor of a compressed Cartesian tensor.

    Args:
        tensor_rank: The rank of the compressed Cartesian tensor.
        trinomial_coefficients: Array of shape (i + 1, j + 1, k + 1) that contains the trinomial coefficients (n, i j k)
        which can be accessed through the array trinomial_coefficients[i, j, k]. The indices of the trinomial
        coefficients i, j, and k have to be at least of the size rank.
    Returns:
        Array with the degeneracies in anti-canonical order.
    """
    tensor_length = (tensor_rank + 1) * (tensor_rank + 2) // 2
    degeneracy_tensor = np.zeros(tensor_length)
    counter = 0
    for i in range(tensor_rank, -1, -1):
        for j in range(tensor_rank - i, -1, -1):
            k = tensor_rank - i - j
            degeneracy_tensor[counter] = trinomial_coefficients[i, j, k]
            counter += 1
    return degeneracy_tensor


def rank(tensor: np.ndarray) -> int:
    """Compute the tensor rank from the length of a compressed Cartesian tensor.

    Args:
        tensor: Array of the compressed Cartesian tensor.

    Returns:
        Rank of the compressed Cartesian tensor.
    """
    tensor_rank = round(0.5 * (np.sqrt(8.0 * len(tensor) + 1.0) - 3.0))
    return tensor_rank


def length(tensor_rank: int) -> int:
    """Compute the length of a compressed Cartesian tensor from the rank of its uncompressed form.

    Args:
        tensor_rank: Rank of the compressed Cartesian tensor.

    Returns:
        Length of the compressed Cartesian tensor.
    """
    return (tensor_rank + 2) * (tensor_rank + 1) // 2


def convert_multi_index(alpha: Tuple[int, int, int]) -> int:
    """Convert a multi-index to the corresponding compressed Cartesian tensor index.

    Args:
        alpha: Multi-index corresponding to the order of x, y, and z Cartesian components.

    Returns:
        Compressed Cartesian tensor index.
    """
    return (alpha[1] ** 2 + 2 * alpha[1] * alpha[2] + alpha[1] + alpha[2] ** 2 + 3 * alpha[2]) // 2 + 1


def convert_tensor_index(tensor_index: int, tensor_rank: int) -> Tuple[int, int, int]:
    """Convert from a compressed Cartesian tensor index to the corresponding multi-index.

    Args:
        tensor_index: Compressed Cartesian tensor index.
        tensor_rank: The rank of the compressed Cartesian tensor.

    Returns:
        Tuple with the multi-index corresponding to the order of x, y, and z Cartesian components.
    """
    i = 1
    for ax in range(tensor_rank, -1, -1):
        for ay in range(tensor_rank - ax, -1, -1):
            az = tensor_rank - ax - ay
            if i == tensor_index:
                return ax, ay, az
            else:
                i += 1


def compute_interaction_tensor_element(multi_index: List[np.ndarray],
                                       distance_vector: np.ndarray,
                                       tensor_coefficients: np.ndarray) -> float:
    """Compute the element of an interaction tensor corresponding to the given multi-index containing the order of the
    derivatives of a distance vector wrt the x, y, and z Cartesian components.

        See C. E. Dykstra, J. Comput. Chem., 9 (1988), 476 for details of the algorithm.

    Args:
        multi_index: Multi-index containing the order of derivatives wrt to x, y, and z Cartesian components of the
        distance vector.
        distance_vector: Distance vector, i.e., r_ba = r_a - r_b
        tensor_coefficients: Array of shape (max_order+1, max_order+1, 2*max_order+2), containing
        the tensor coefficients.

    Returns:
        Element of the interaction tensor corresponding to a given multi-index and distance vector.
    """
    tensor_element = 0.0
    i, j, k = multi_index[0] + multi_index[1]
    norm = np.linalg.norm(distance_vector)
    for q in range(i + 1):
        cl = tensor_coefficients[q, i, 1] * (distance_vector[0] / norm) ** q
        o = q + i + 1
        for m in range(j + 1):
            cm = cl * tensor_coefficients[m, j, o] * (distance_vector[1] / norm) ** m
            p = o + j + m
            for n in range(k + 1):
                cn = cm * tensor_coefficients[n, k, p] * (distance_vector[2] / norm) ** n
                tensor_element += cn
    tensor_element /= norm ** (i + j + k + 1)
    return tensor_element * (-1) ** (multi_index[1][0] + multi_index[1][1] + multi_index[1][2])


def compute_tensor_coefficients(max_order: int) -> np.ndarray:
    """Compute tensor coefficients needed in the compute_tensor_element function.

       See C. E. Dykstra, J. Comput. Chem., 9 (1988), 476, where C^(n)_ij are stored as tensor_coefficients(j,i,n).

    Args:
        max_order: An integer representing the maximum order of the tensor coefficient.

    Returns:
        Array of shape (max_order+1, max_order+1, 2*max_order+2), containing the tensor coefficients.
    """
    tensor_coefficients = np.zeros((max_order + 1, max_order + 1, 2 * max_order + 2))
    tensor_coefficients[0, 0, :] = 1.0
    for n in range(1, 2 * max_order + 2):
        if n % 2 == 0:
            continue
        for i in range(1, max_order + 1):
            if i % 2 == 0:
                k = i
            else:
                k = i - 1
            for j in range(0, i + 1):
                if (i + j) % 2 != 0:
                    continue
                if j == 0:
                    tensor_coefficients[j, i, n] = tensor_coefficients[j + 1, i - 1, n]
                elif j != i:
                    tensor_coefficients[j, i, n] = (j + 1.0) * tensor_coefficients[j + 1, i - 1, n]
                    tensor_coefficients[j, i, n] -= (n + k) * tensor_coefficients[j - 1, i - 1, n]
                    k += 2
                else:
                    tensor_coefficients[j, i, n] = -(n + k) * tensor_coefficients[j - 1, i - 1, n]
    return tensor_coefficients


def compute_trace(tensor: np.ndarray, alpha: Tuple[int, int, int], trinomial_coefficients: np.ndarray) -> float:
    """Compute the trace of a compressed Cartesian tensor along a given axis through the multi-index notation.

    Args:
        tensor: Array of the compressed Cartesian tensor.
        alpha: Multi-index corresponding to the order of x, y, and z Cartesian components.
        trinomial_coefficients: Array containing the trinomial coefficients up to their given order. The order needs to
        be at least 1.

    Returns:
        A float representing the trace of the tensor computed by the function.
    """
    tensor_trace = 0.0
    for bx in range(1, -1, -1):
        for by in range(1 - bx, -1, -1):
            bz = 1 - bx - by
            ti = convert_multi_index((alpha[0] + 2 * bx, alpha[1] + 2 * by, alpha[2] + 2 * bz))
            tensor_trace += trinomial_coefficients[bx, by, bz] * tensor[ti - 1]
    return tensor_trace


def detrace(tensor: np.ndarray, factorials: np.ndarray, double_factorials: np.array,
            trinomial_coefficients: np.ndarray) -> np.ndarray:
    """Remove the trace of a compressed Cartesian tensor.

    Args:
        tensor: Array of the compressed Cartesian tensor.
        factorials: Array that contains the factorials of their given order. The factorials are
        stored in ascending order. The order needs to be at least (tensor_rank) for the tensor_rank of the tensor to be
        detraced.
        double_factorials: Array that contains the double factorials of their given order. The double factorials are
        stored in ascending order. The order needs to be at least (2 * tensor_rank - 1) for the tensor_rank of the
        tensor to be detraced.
        trinomial_coefficients: Array containing the trinomial coefficients up to their given order. The order needs to
        be at least 1.

    Returns:
        Array of the detraced compressed Cartesian tensor.
    """
    tensor_rank = rank(tensor)
    divisor = double_factorials[(2 * tensor_rank - 1)]
    trace_tensor = np.zeros(len(tensor))
    for ax in range(tensor_rank, -1, -1):
        for ay in range(tensor_rank - ax, -1, -1):
            az = tensor_rank - ax - ay
            ti = convert_multi_index((ax, ay, az))
            for bx in range(0, int(float(ax) / 2.0) + 1):
                xn = factorials[ax] / (
                        2.0 ** bx * factorials[bx] * factorials[ax - 2 * bx])
                for by in range(0, int(float(ay) / 2.0) + 1):
                    yn = factorials[ay] / (
                            2.0 ** by * factorials[by] * factorials[(ay - 2 * by)])
                    for bz in range(0, int(float(az) / 2.0) + 1):
                        zn = factorials[az] / (
                                2.0 ** bz * factorials[bz] * factorials[az - 2 * bz])
                        norm = bx + by + bz
                        if norm != 1:
                            continue
                        sgn = (-1.0) ** norm
                        trace_tensor[ti - 1] += (sgn * double_factorials[
                            (2 * tensor_rank - 2 * norm - 1)] * xn * yn * zn
                                                 * compute_trace(tensor, (ax - 2 * bx, ay - 2 * by, az - 2 * bz),
                                                                 trinomial_coefficients)) / divisor
    detraced_tensor = tensor + trace_tensor
    return detraced_tensor


def uncompress_symmetric_matrix(a : np.ndarray
                                ) -> np.ndarray:
    """Uncompresses a compressed symmetric matrix.

    Args:
         a: Compressed symmetric matrix.

    Returns:
        Uncompressed symmetric matrix
    """
    n = int(np.sqrt(a.size * 2))
    if n * (n + 1) != a.size * 2:
        raise ValueError("Invalid size for compressed symmetric matrix.")
    r, c = np.triu_indices(n)
    out = np.zeros((n, n))
    out[r, c] = a
    out[c, r] = a
    return out
