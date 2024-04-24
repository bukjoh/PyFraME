// Global variables

#ifndef _global_h_
#define _global_h_

#include <Eigen/Dense>
#include <vector>
#include <unordered_set>

namespace global {
//global variables required for general functionality
//these are intended to be set once (or a few times) before use and
extern int rank;
extern int max_order;
extern std::vector<Eigen::MatrixXd> tensor_coefficients;
extern Eigen::Matrix<Eigen::Matrix<int, 2, 3>, Eigen::Dynamic, Eigen::Dynamic> tensor_template_interaction;
extern Eigen::Matrix<Eigen::Matrix<int, 2, 3>, Eigen::Dynamic, Eigen::Dynamic> tensor_template_potential;

//global variables required for IndDipolesFields, need to be
extern std::vector<Eigen::Vector3d> coordinates;
extern Eigen::VectorXi indices;
extern std::vector<std::unordered_set<int>> exclusions;
extern Eigen::MatrixXd old_ind_dipoles;

//global variables required for NuclearFields
extern std::vector<Eigen::Vector3d> nuclei_coordinates;
extern Eigen::VectorXd nuclei_charges;
//global variables required for MultipoleFields
extern Eigen::VectorXi multipole_orders;
extern std::vector<Eigen::VectorXd> multipoles;
// global variables required for perturbed and unperturbed VdW-Interaction
extern Eigen::VectorXd quantum_sigmas;
extern Eigen::VectorXd quantum_epsilons;
extern Eigen::VectorXd classical_sigmas;
extern Eigen::VectorXd classical_epsilons;
extern std::string combination_rule;
extern Eigen::VectorXd factorials;
}

#endif