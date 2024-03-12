#ifndef _computation_h_
#define _computation_h_

#include <Eigen/Dense>
#include <omp.h>
#include <unordered_set>
#include <cmath>
#include <vector>

namespace computation
{
//Calculates the length of a polytensor given start_rank and end_rank in that dimension.
int get_polytensor_length(int start_rank, int end_rank);

//Computes an interaction tensor element from a given multiindex, r_ab == r_b - r_a, and tensor_coefficients
double compute_interaction_tensor_element(
    const Eigen::Matrix<int, 2, 3> &multiindex,
    const Eigen::Vector3d &r_ab,
    const std::vector<Eigen::MatrixXd> &tensor_coefficients);

//Computes the t_tensor for the interaction of two atoms in a specified range of indices.
Eigen::MatrixXd compute_t_tensor(
    const Eigen::Vector3d &r_ab,
    const Eigen::Matrix<Eigen::Matrix<int, 2, 3>, Eigen::Dynamic, Eigen::Dynamic> &tensor_template,
    const std::vector<Eigen::MatrixXd> &tensor_coefficients,
    int rank_a,
    int rank_b,
    int start_rank_a,
    int start_rank_b);
}

#endif