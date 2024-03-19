#include "global.h"

namespace global {
//global variables required for general functionality
//these are intended to be set once (or a few times) before use and
int rank = -1;
int max_order = -1;
std::vector<Eigen::MatrixXd> tensor_coefficients;
Eigen::Matrix<Eigen::Matrix<int, 2, 3>, Eigen::Dynamic, Eigen::Dynamic> tensor_template_interaction;
Eigen::Matrix<Eigen::Matrix<int, 2, 3>, Eigen::Dynamic, Eigen::Dynamic> tensor_template_potential;

//global variables required for IndDipolesFields, need to be
std::vector<Eigen::Vector3d> coordinates;
Eigen::VectorXi indices;
std::vector<std::unordered_set<int>> exclusions;
Eigen::MatrixXd old_ind_dipoles;
//global variables required for NuclearFields
Eigen::VectorXd nuclei_charges;
std::vector<Eigen::Vector3d> nuclei_coordinates;
//global variables required for MultipoleFields
Eigen::VectorXi multipole_orders;
std::vector<Eigen::VectorXd> multipoles;


}