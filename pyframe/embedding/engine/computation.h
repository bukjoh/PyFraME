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

Eigen::MatrixXd compute_perturbed_t_tensor(
    const Eigen::Vector3d &r_ab,
    const Eigen::Matrix<Eigen::Matrix<int, 2, 3>, Eigen::Dynamic, Eigen::Dynamic> &tensor_template,
    const Eigen::Matrix<int, 2, 3> &perturbation_tuple,
    int rank_a,
    int rank_b,
    int start_rank_a,
    int start_rank_b);

double e_nuc_es_perturbed(
       int start,
       int end,
       int nuc_idx,
       const Eigen::Matrix<int, 2, 3> &perturbation_tuple);

// Computes the field caused by induced dipoles at site i.
// Parallelized with OpenMP.
Eigen::MatrixXd ind_dipoles_field(int start, int end);

// Computes the field of the nuclei on all atoms
// Parallelized with OpenMP.
Eigen::MatrixXd nuclei_fields(int start, int end);

// Computes the field gradients of the nuclei on all atoms
// Parallelized with OpenMP.
std::vector<Eigen::MatrixXd> nuclei_field_gradients(int start, int end);

// Computes the field of the multipoles on a the ith coordinates
// Parallelized with OpenMP.
Eigen::MatrixXd multipole_field(int i);

// Computes self energy for given array of indexes
// Parallelized with OpenMP.
double environment_energy(Eigen::MatrixXi idx_list);

// Computes the energy between all atoms and the nuclei.
// Parallelized with OpenMP.
double e_nuc_es(int start, int end);

// Uses the Lorentz-Berthelot combination rules for non-bonded VdW interactions.
std::tuple<double, double> LB_combination(double sigma_i, double sigma_j, double epsilon_i, double epsilon_j);

// Computes the LJ repulsion potential between a ClassicalSubsystem and a QuantumSubsystem.
// Parallelized with OpenMP.
double compute_unperturbed_lj_repulsion(int start, int end, std::string combination_rule);

// Computes the LJ dispersion potential between a ClassicalSubsystem and a QuantumSubsystem.
// Parallelized with OpenMP.
double compute_unperturbed_lj_dispersion(int start, int end, std::string combination_rule);

// Computes the LJ repulsion gradients of the Nuclei in a QuantumSubsystem interacting with a ClassicalSubsystem.
// Parallelized with OpenMP.
std::vector<Eigen::Vector3d> compute_lj_repulsion_gradient(int start, int end, std::string combination_rule);

// Computes the LJ dispersion gradients of the Nuclei in a QuantumSubsystem interacting with a ClassicalSubsystem.
// Parallelized with OpenMP.
std::vector<Eigen::Vector3d> compute_lj_dispersion_gradient(int start, int end, std::string combination_rule);

// Computes the derivative of the LJ repulsion potential between a ClassicalSubsystem and a nucleus.
// Parallelized with OpenMP.
double compute_perturbed_lj_repulsion(int start,
                                      int end,
                                      int nuc_idx,
                                      std::string combination_rule,
                                      std::vector<std::vector<std::vector<Eigen::Matrix<int, 2, 3>>>> k_partitions);

// Computes the derivative of the LJ dispersion potential between a ClassicalSubsystem and a nucleus.
// Parallelized with OpenMP.
double compute_perturbed_lj_dispersion(int start,
                                      int end,
                                      int nuc_idx,
                                      std::string combination_rule,
                                      std::vector<std::vector<std::vector<Eigen::Matrix<int, 2, 3>>>> k_partitions);

}

#endif