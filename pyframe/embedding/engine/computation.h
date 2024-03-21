#ifndef _computation_h_
#define _computation_h_

#include "global.h"
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
    int rank_a,
    int rank_b,
    int start_rank_a,
    int start_rank_b);

// Computes the field caused by induced dipoles at site i.
// Parallelized with OpenMP.
Eigen::MatrixXd ind_dipoles_field(int start, int end);

// Computes the field of the nuclei on all atoms
// Parallelized with OpenMP.
Eigen::MatrixXd nuclei_fields(int start, int end);

// Computes the field of the multipoles on a the ith coordinates
// Parallelized with OpenMP.
Eigen::MatrixXd multipole_field(int i);

// Computes self energy for given array of indexes
// Parallelized with OpenMP.
double self_energy(Eigen::MatrixXi idx_list);

// Computes the energy between all atoms and the nuclei.
// Parallelized with OpenMP.
double e_nuc_es(int start, int end);

}

#endif