from __future__ import annotations

import numpy as np
from typing import Optional
from pyframe.embedding import tensor_tools


class FirstDegreePolytensor:
    """A FirstDegreePolytensor object describes a compressed first degree cartesian polytensor and its operations. A
    compressed first degree polytensor is a set of compressed Cartesian tensors in a sequence of increasing rank.

    See Jon Applequist J. Math. Phys. 24, 736 (1983) for details.

    Args:
        rank: Rank of the compressed Cartesian tensor of the highest rank in the polytensor.
        data_type: Datatype of the elements in the polytensor.
        tensor_data: Dataset of compressed Cartesian tensors in a sequence of increasing rank that will be written into
        the polytensor.
    """

    def __init__(self,
                 rank: int,
                 data_type=float,
                 tensor_data: Optional[np.ndarray] = None
                 ):
        self._rank = rank
        self.length = (rank + 1) * (rank + 2) * (rank + 3) // 6
        if tensor_data is not None:
            self.data = tensor_data
        else:
            self.data = np.zeros(self.length, dtype=data_type)

    def write_to_data_block_wise(self, compressed_tensor: np.array) -> None:
        """Writes compressed tensor to data based on its length."""
        start_idx = 0
        for n in range(self._rank + 1):
            if (n + 1) * (n + 2) // 2 == len(compressed_tensor):
                break
            start_idx += (n + 1) * (n + 2) // 2

        start_idx = int(start_idx)
        self.data[start_idx: start_idx + len(compressed_tensor)] = compressed_tensor[:]

    def write_to_data(self, i, new_data):
        """Writes new data to data

        Args:
             i: Column index of data.
             new_data: New data written to data.
        """
        self.data[i] = new_data

    def __add__(self, other) -> FirstDegreePolytensor:
        """Sum of two FirstDegreePolytensors.

        Returns:
            Elementwise addition.
        """
        return FirstDegreePolytensor(rank=self._rank, tensor_data=np.add(self.data, other.data))

    def multiply_elementwise(self, other) -> FirstDegreePolytensor:
        """Multiplication of two FirstDegreePolytensors.

        Returns:
            Elementwise product.
        """
        return FirstDegreePolytensor(rank=self._rank, tensor_data=np.multiply(self.data, other.data))

    def multiply_scalar_matrix(self, other) -> np.array:
        """Contraction between a FirstDegreePolytensors of scalars (A) on the left and FirstDegreePolytensor of
        Matrices (B) on the right.

        Returns:
            Vector-Vector dot-product (B.T@A).
        """
        return np.einsum('i, ijk', self.data, other.data)

    def multiply_matrix_scalar(self, other) -> np.array:
        """Contraction between a FirstDegreePolytensors of scalars (A) on the right and FirstDegreePolytensor of
        Matrices (B) on the left.

        Returns:
            Vector-Vector dot-product (A.T@B).
        """
        return other.multiply_scalar_matrix(self)

    def truncate_tensor(self, order: int) -> FirstDegreePolytensor:
        """Truncate sequence of compressed Cartesian tensors.

        Args:
            order: Highest rank of the compressed Cartesian tensor until which the FirstDegreePolytensor is going to be
            truncated.

        Returns:
            Truncated tensor.
        """
        end = (order + 1) * (order + 2) * (order + 3) // 6
        return FirstDegreePolytensor(rank=order, tensor_data=self.data[:end])

    def multiply_first_degree_second_degree(self, other) -> FirstDegreePolytensor:
        """Contraction between a FirstDegreePolytensor (A) and a SecondDegreePolytensor (B).

        Returns:
            Vector-Matrix product between a (A) on the left and (B) on the right (A.T@B).
        """
        return FirstDegreePolytensor(rank=self._rank, tensor_data=np.einsum('i, ij -> j', self.data, other.data))

    def dot_first_degree(self, other) -> np.array:
        """Contraction between two FirstDegreePolytensors of scalars (A).

        Returns:
            Vector-Vector dot-product (A.T@A).
        """
        return np.einsum('i, i', self.data, other.data)


class SecondDegreePolytensor:
    """A SecondDegreePolytensor object describes a compressed second degree cartesian polytensor and its operations. A
    compressed second degree polytensor is represented by a rectangular matrix whose blocks are tensors whose rank is
    subdivided into two indices.

    See Jon Applequist J. Math. Phys. 24, 736 (1983) for details.

    Args:
        rank_1: Rank of the compressed Cartesian tensor of the highest rank in the first column of the polytensor.
        rank_2: Rank of the compressed Cartesian tensor of the highest rank in the first row of the
        polytensor. If rank_2 is not defined it will be set to rank_1.
        * (rank_2 + 3) // 6 to (rank_2 + 1) * (rank_2 + 2) // 2.
        data_type: Datatype of the elements in the polytensor.
        tensor_data: Dataset that will be written into the polytensor.
    """

    def __init__(self,
                 rank_1: int | list,
                 rank_2: int | list,
                 data_type=float,
                 tensor_data: Optional[np.ndarray] = None
                 ):
        if isinstance(rank_1, list):
            self.length_1 = 0
            for i in range(rank_1[0], rank_1[-1] + 1):
                self.length_1 += (i + 1) * (i + 2) // 2
            self._rank_1 = rank_1[-1]
        else:
            self._rank_1 = rank_1
            self.length_1 = (rank_1 + 1) * (rank_1 + 2) * (rank_1 + 3) // 6
        if isinstance(rank_2, list):
            self.length_2 = 0
            for i in range(rank_2[0], rank_2[-1] + 1):
                self.length_2 += (i + 1) * (i + 2) // 2
            self._rank_2 = rank_2[-1]
        else:
            self._rank_2 = rank_2
            self.length_2 = (rank_2 + 1) * (rank_2 + 2) * (rank_2 + 3) // 6
        if tensor_data is not None:
            self.data = tensor_data
        else:
            self.data = np.zeros([self.length_1, self.length_2], dtype=data_type)

    def write_to_data(self, i, j, new_data):
        """Writes new data to data.

        Args:
             i: row index of data.
             j: column index of data.
             new_data: New data written to data.
        """
        self.data[i, j] = new_data

    def write_interaction_tensor_multi_indices(self):
        """Writes the interaction tensor multi-indices to data that can be used to calculate the corresponding
        interaction tensor elements."""
        idx = 0
        for i in range(self._rank_1 + 1):
            for j in range((i + 1) * (i + 2) // 2):
                tensor_idx = np.asarray(tensor_tools.convert_tensor_index(tensor_index=j + 1, tensor_rank=i))
                self.data[0][idx] = [np.array([0, 0, 0]), tensor_idx]
                self.data[idx][0] = [tensor_idx, np.array([0, 0, 0])]
                idx += 1
        for i in range(1, self.length_1):
            for j in range(1, self.length_2):
                self.data[i][j] = [self.data[i][0][0], self.data[0][j][1]]

    def write_potential_tensor_multi_indices(self):
        """Writes the interaction tensor multi-indices to data that can be used to calculate the corresponding
        interaction tensor elements."""
        idx = 0
        for i in range(self._rank_1 + 1):
            for j in range((i + 1) * (i + 2) // 2):
                tensor_idx = np.asarray(tensor_tools.convert_tensor_index(tensor_index=j + 1, tensor_rank=i))
                self.data[0][idx] = [tensor_idx, np.array([0, 0, 0])]
                self.data[idx][0] = [tensor_idx, np.array([0, 0, 0])]
                idx += 1
        for i in range(1, self.length_1):
            for j in range(1, self.length_2):
                self.data[i][j] = [self.data[i][0][0] + self.data[0][j][0], np.array([0, 0, 0])]

    def multiply_second_degree_first_degree(self, other) -> FirstDegreePolytensor:
        """Contraction between a SecondDegreePolytensor (A) and a FirstDegreePolytensor (B).

        Returns:
            Matrix-Vector product between a (A) on the left and (B) on the right (A@B).
        """
        return FirstDegreePolytensor(rank=self._rank_2, tensor_data=np.einsum('ij, j -> i', self.data, other.data))
