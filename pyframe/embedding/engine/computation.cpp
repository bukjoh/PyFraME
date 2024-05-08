#include "computation.h"
#include <iostream>


namespace computation
{
//Calculates the length of a polytensor given start_rank and end_rank in that dimension.
int get_polytensor_length(int start_rank, int end_rank) {
    int length = 0;
    for(int i = start_rank; i <= end_rank; i++) {
        length += (i + 1) * (i + 2) / 2;
    }
    return length;
}

//Computes an interaction tensor element from a given multiindex, r_ab == r_b - r_a, and tensor_coefficients
double compute_interaction_tensor_element(
    const Eigen::Matrix<int, 2, 3> &multiindex,
    const Eigen::Vector3d &r_ab,
    const std::vector<Eigen::MatrixXd> &tensor_coefficients) {
    int i = multiindex.col(0).sum(), j = multiindex.col(1).sum(), k = multiindex.col(2).sum();
    double element = 0;

    // TODO: divide r_ab by norm before the loops to optimize performance
    double norm = r_ab.norm();

    for (int q = 0; q <= i; q++) {
        double cl = tensor_coefficients[q](i, 1) * std::pow(r_ab(0) / norm, q);
        int o = q + i + 1;
        for (int m = 0; m <= j; m++) {
            double cm = cl * tensor_coefficients[m](j, o) * std::pow(r_ab(1) / norm, m);
            int p = o + j + m;
            for (int n = 0; n <= k; n++) {
                double cn = cm * tensor_coefficients[n](k, p) * std::pow(r_ab(2) / norm, n);
                element += cn;
            }
        }
    }
    element /= std::pow(norm, i + j + k + 1);
    element *= std::pow(-1, multiindex.row(1).sum());
    return element;
}

//Computes the t_tensor for the interaction of two atoms in a specified range of indices.
Eigen::MatrixXd compute_t_tensor(
    const Eigen::Vector3d &r_ab,
    const Eigen::Matrix<Eigen::Matrix<int, 2, 3>, Eigen::Dynamic, Eigen::Dynamic> &tensor_template,
    int rank_a,
    int rank_b,
    int start_rank_a,
    int start_rank_b) {
    Eigen::MatrixXd interaction_tensor(get_polytensor_length(start_rank_a, rank_a),
                                        get_polytensor_length(start_rank_b, rank_b));
    int start_b = (start_rank_b) * (start_rank_b + 1) * (start_rank_b + 2) / 6;
    int end_b = (rank_b + 1) * (rank_b + 2) * (rank_b + 3) / 6;
    int start_a = (start_rank_a) * (start_rank_a + 1) * (start_rank_a + 2) / 6;
    int end_a = (rank_a + 1) * (rank_a + 2) * (rank_a + 3) / 6;

    for(int i = start_a; i < end_a; i++) {
        for(int j = start_b; j < end_b; j++) {
            double interaction_element = compute_interaction_tensor_element(
                tensor_template(i, j), r_ab, global::tensor_coefficients);
            interaction_tensor(i - start_a, j - start_b) = interaction_element;
        }
    }
    return interaction_tensor;
}

//Computes the t_tensor for the interaction of two atoms in a specified range of indices.
Eigen::MatrixXd compute_perturbed_t_tensor(
    const Eigen::Vector3d &r_ab,
    const Eigen::Matrix<Eigen::Matrix<int, 2, 3>, Eigen::Dynamic, Eigen::Dynamic> &tensor_template,
    const Eigen::Matrix<int, 2, 3> &perturbation_tuple,
    int rank_a,
    int rank_b,
    int start_rank_a,
    int start_rank_b) {
    Eigen::MatrixXd interaction_tensor(get_polytensor_length(start_rank_a, rank_a),
                                        get_polytensor_length(start_rank_b, rank_b));
    int start_b = (start_rank_b) * (start_rank_b + 1) * (start_rank_b + 2) / 6;
    int end_b = (rank_b + 1) * (rank_b + 2) * (rank_b + 3) / 6;
    int start_a = (start_rank_a) * (start_rank_a + 1) * (start_rank_a + 2) / 6;
    int end_a = (rank_a + 1) * (rank_a + 2) * (rank_a + 3) / 6;

    for(int i = start_a; i < end_a; i++) {
        for(int j = start_b; j < end_b; j++) {
            double interaction_element = compute_interaction_tensor_element(
                tensor_template(i, j) + perturbation_tuple, r_ab, global::tensor_coefficients);
            interaction_tensor(i - start_a, j - start_b) = interaction_element;
        }
    }
    return interaction_tensor;
}


// Computes the field caused by induced dipoles at atom i.
// Parallelized with OpenMP.
Eigen::MatrixXd ind_dipoles_field(int start, int end) {
    int no_atoms = static_cast<int>(global::atom_coordinates.size());
    Eigen::MatrixXd ind_dipoles_field = Eigen::MatrixXd::Zero(no_atoms, 3);
    for(int i = start; i < end; i++) {
        #pragma omp for
        for(int j = 0; j < no_atoms; j++) {
            if(global::exclusions[i].find(global::indices[j]) != global::exclusions[i].end()) {
                continue;
            }
            Eigen::Vector3d r_ab = global::atom_coordinates[i] - global::atom_coordinates[j];
            Eigen::MatrixXd t_tensor = compute_t_tensor(r_ab,
                                                        global::tensor_template_potential,
                                                        1, 1, 1, 1);
            #pragma omp critical
            ind_dipoles_field.row(i) += (t_tensor * global::old_ind_dipoles.row(j).transpose()).transpose();
        }
    }
    return ind_dipoles_field;
}


// Computes the field caused by induced dipoles at atom i.
// Parallelized with OpenMP.
Eigen::MatrixXd target_source_ind_dipoles_field(Eigen::VectorXi targets, Eigen::VectorXi sources) {
    Eigen::MatrixXd ind_dipoles_field = Eigen::MatrixXd::Zero(targets.size(), 3);
    for (int i = 0; i < targets.size(); i++) {
        #pragma omp for
        for(int j = 0; j < sources.size(); j++) {
            if(global::exclusions[targets[i]].find(global::indices[sources[j]]) != global::exclusions[targets[i]].end()) {
                continue;
            }
            Eigen::Vector3d r_ab = global::atom_coordinates[targets[i]] - global::atom_coordinates[sources[j]];
            Eigen::MatrixXd t_tensor = compute_t_tensor(r_ab,
                                                        global::tensor_template_potential,
                                                        1, 1, 1, 1);
            #pragma omp critical
            ind_dipoles_field.row(targets[i]) += (t_tensor * global::old_ind_dipoles.row(sources[j]).transpose()).transpose();
        }
    }
    return ind_dipoles_field;
}

// Computes the field of the nuclei on all atoms
// Parallelized with OpenMP.
Eigen::MatrixXd nuclei_fields(int start, int end) {
    int no_nuclei = static_cast<int>(global::nuclei_coordinates.size());
    int no_atoms = static_cast<int>(global::coordinates.size());

    Eigen::MatrixXd nuclei_fields = Eigen::MatrixXd::Zero(no_atoms, 3);
    #pragma omp parallel
    {
        Eigen::MatrixXd field_part = Eigen::MatrixXd::Zero(no_atoms, 3);
        #pragma omp for
        for(int i = start; i < end; i++) {
            for(int j = 0; j < no_nuclei; j++) {
                Eigen::Vector3d r_ab = global::coordinates[i] - global::nuclei_coordinates[j];
                Eigen::MatrixXd t_tensor = compute_t_tensor(r_ab,
                                                            global::tensor_template_potential,
                                                            0, 1, 0, 1);
                field_part.row(i) += t_tensor * global::nuclei_charges(j);
            }
        }
        #pragma omp critical
        {
            nuclei_fields += field_part;
        }
    }
    return nuclei_fields;
}

// Computes the field of the nuclei on all atoms
// Parallelized with OpenMP.
std::vector<Eigen::MatrixXd> nuclei_field_gradients(int start, int end) {
    const int no_nuclei = static_cast<int>(global::nuclei_coordinates.size());
    const int no_atoms = static_cast<int>(global::coordinates.size());
    std::vector<Eigen::MatrixXd> nuclei_field_gradients(no_nuclei, Eigen::MatrixXd::Zero(no_atoms, 6));
    #pragma omp parallel
    {
        std::vector<Eigen::MatrixXd> local_nucleus_field_gradients(no_nuclei, Eigen::MatrixXd::Zero(no_atoms, 6));

        #pragma omp for
        for(int i = start; i < end; i++) {
            for(int j = 0; j < no_nuclei; j++) {
                Eigen::Vector3d r_ab = global::coordinates[i] - global::nuclei_coordinates[j];
                Eigen::MatrixXd t_tensor = compute_t_tensor(r_ab,
                                                            global::tensor_template_potential,
                                                            0, 2, 0, 2);
                local_nucleus_field_gradients[j].row(i) += t_tensor * global::nuclei_charges(j);
            }
        }
        #pragma omp critical
        {
            // Combine thread-local nucleus_field_gradient_part to the final nuclei_field_gradients
            for (int k = 0; k < no_nuclei; ++k) {
                nuclei_field_gradients[k] += local_nucleus_field_gradients[k];
            }
        }
    }
    return nuclei_field_gradients;
}

// Computes the field caused by all atoms at atom i.
// Parallelized with OpenMP.
Eigen::MatrixXd multipole_field(int i) {
    Eigen::MatrixXd multipole_field = Eigen::MatrixXd::Zero(3, 1);
    int no_atoms = static_cast<int>(global::atom_coordinates.size());
    #pragma omp parallel
    {
        Eigen::MatrixXd field_part = Eigen::MatrixXd::Zero(3, 1);
        #pragma omp for
        for(int j = 0; j < no_atoms; j++) {
            if(global::exclusions[i].find(global::indices[j]) != global::exclusions[i].end()) {
                continue;
            }
            Eigen::Vector3d r_ab = global::atom_coordinates[i] - global::atom_coordinates[j];
            Eigen::MatrixXd t_tensor = compute_t_tensor(r_ab,
                                                        global::tensor_template_potential,
                                                        global::multipole_orders[j], 1, 0, 1);
            field_part += (global::multipoles[j].transpose() * t_tensor).transpose() ;
        }
        #pragma omp critical
        {
            multipole_field += field_part;
        }
    }

    return multipole_field;
}

// Computes the self energy of a ClassicalSubsystem for given array of indexes.
// Parallelized with OpenMP.
double self_energy(Eigen::MatrixXi idx_arr) {
    double self_energy = 0.0;
    #pragma omp parallel
    {
        double energy_contr = 0.0;
        #pragma omp for
        for(int k = 0; k < idx_arr.cols(); k++){
            int i = idx_arr(0, k);
            int j = idx_arr(1, k);
            if(global::exclusions[i].find(global::indices[j]) != global::exclusions[i].end()) {
                continue;
            }
            Eigen::Vector3d r_ab = global::atom_coordinates[i] - global::atom_coordinates[j];
            Eigen::MatrixXd t_tensor = compute_t_tensor(r_ab,
                                                        global::tensor_template_interaction,
                                                        global::multipole_orders[j], global::multipole_orders[i], 0, 0);
            energy_contr += global::multipoles[j].transpose() * t_tensor * global::multipoles[i];
        }
    #pragma omp critical
    {
    self_energy += energy_contr;
    }
    }
    return self_energy;
}

// Computes the energy between all atoms and the nuclei.
// Parallelized with OpenMP.
double e_nuc_es(int start, int end) {
    double e_nuc_es = 0.0;
    int no_nuclei = static_cast<int>(global::nuclei_coordinates.size());
    #pragma omp parallel
    {
        double energy_contr = 0.0;
        #pragma omp for
        for(int j = start; j < end; j++){
            for(int i = 0; i < no_nuclei; i++) {
                Eigen::Vector3d r_ab = global::nuclei_coordinates[i] - global::coordinates[j];
                Eigen::MatrixXd t_tensor = compute_t_tensor(r_ab,
                                                            global::tensor_template_interaction,
                                                            global::multipole_orders[j], 0, 0, 0);
                energy_contr += (global::multipoles[j].transpose() * t_tensor * global::nuclei_charges[i])[0];
            }
        }
        #pragma omp critical
        {
            e_nuc_es += energy_contr;
        }
    }
    return e_nuc_es;
}

// Computes the energy between all atoms and the nuclei.
// Parallelized with OpenMP.
double e_nuc_es_perturbed(int start, int end, int nuc_idx, const Eigen::Matrix<int, 2, 3> &perturbation_tuple) {
    double e_nuc_es = 0.0;
    #pragma omp parallel
    {
        double energy_contr = 0.0;
        #pragma omp for
        for(int j = start; j < end; j++){
            Eigen::Vector3d r_ab = global::nuclei_coordinates[nuc_idx] - global::coordinates[j];
            Eigen::MatrixXd t_tensor = compute_perturbed_t_tensor(r_ab,
                                                                  global::tensor_template_interaction,
                                                                  perturbation_tuple,
                                                                  global::multipole_orders[j], 0, 0, 0);
            energy_contr += (global::multipoles[j].transpose() * t_tensor * global::nuclei_charges[nuc_idx])[0];
        }
        #pragma omp critical
        {
            e_nuc_es += energy_contr;
        }
    }
    return e_nuc_es;
}

// Uses the Lorentz-Berthelot combination rules for non-bonded VdW interactions.
std::tuple<double, double> LB_combination(double sigma_i, double sigma_j, double epsilon_i, double epsilon_j) {
    double comb_sigma = 0.5 * (sigma_i + sigma_j);
    double comb_epsilon = std::sqrt(epsilon_i * epsilon_j);
    return std::make_tuple(comb_sigma, comb_epsilon);
}


// Computes the LJ dispersion potential between a ClassicalSubsystem and a QuantumSubsystem.
// Parallelized with OpenMP.
double compute_unperturbed_lj_dispersion(int start, int end, std::string combination_rule) {
    double unperturbed_lj_dispersion = 0.0;
    // Define the function pointer type
    std::tuple<double, double> (*combination_func)(double, double, double, double);
    if (combination_rule == "Lorentz-Berthelot") {
        combination_func = LB_combination;
    }
    int no_nuclei = static_cast<int>(global::quantum_sigmas.size());
    #pragma omp parallel
    {
        double energy_contr = 0.0;
        #pragma omp for
        for(int j = start; j < end; j++){
            for(int i = 0; i < no_nuclei; i++) {
                std::tuple<double, double> combined_sigma_epsilon = (*combination_func)(global::quantum_sigmas[i],
                                                                                        global::classical_sigmas[j],
                                                                                        global::quantum_epsilons[i],
                                                                                        global::classical_epsilons[j]);
                double sigma = std::get<0>(combined_sigma_epsilon);
                double epsilon = std::get<1>(combined_sigma_epsilon);
                double recip_distance = compute_t_tensor((global::nuclei_coordinates[i] - global::coordinates[j]),
                                                         global::tensor_template_potential,
                                                         0, 0, 0, 0)(0,0);
                energy_contr += epsilon * std::pow(sigma, 6) * std::pow(recip_distance, 6);
            }
        }
        #pragma omp critical
        {
            unperturbed_lj_dispersion += energy_contr;
        }
    }
    return (-1.0) * 4.0 * unperturbed_lj_dispersion;
}

// Computes the LJ repulsion potential between a ClassicalSubsystem and a QuantumSubsystem.
// Parallelized with OpenMP.
double compute_unperturbed_lj_repulsion(int start, int end, std::string combination_rule) {
    double unperturbed_lj_repulsion = 0.0;
    // Define the function pointer type
    std::tuple<double, double> (*combination_func)(double, double, double, double);
    if (combination_rule == "Lorentz-Berthelot") {
        combination_func = LB_combination;
    }
    int no_nuclei = static_cast<int>(global::quantum_sigmas.size());
    #pragma omp parallel
    {
        double energy_contr = 0.0;
        #pragma omp for
        for(int j = start; j < end; j++){
            for(int i = 0; i < no_nuclei; i++) {
                std::tuple<double, double> combined_sigma_epsilon = (*combination_func)(global::quantum_sigmas[i],
                                                                                        global::classical_sigmas[j],
                                                                                        global::quantum_epsilons[i],
                                                                                        global::classical_epsilons[j]);
                double sigma = std::get<0>(combined_sigma_epsilon);
                double epsilon = std::get<1>(combined_sigma_epsilon);
                double recip_distance = compute_t_tensor((global::nuclei_coordinates[i] - global::coordinates[j]),
                                                         global::tensor_template_potential,
                                                         0, 0, 0, 0)(0,0);
                energy_contr += epsilon * std::pow(sigma, 12) * std::pow(recip_distance, 12);
            }
        }
        #pragma omp critical
        {
            unperturbed_lj_repulsion += energy_contr;
        }
    }
    return 4.0 * unperturbed_lj_repulsion;
}

// Computes the LJ repulsion gradients of the Nuclei in a QuantumSubsystem interacting with a ClassicalSubsystem.
// Parallelized with OpenMP.
std::vector<Eigen::Vector3d> compute_lj_repulsion_gradient(int start, int end, std::string combination_rule) {
    // Define the function pointer type
    std::tuple<double, double> (*combination_func)(double, double, double, double);
    if (combination_rule == "Lorentz-Berthelot") {
        combination_func = LB_combination;
    }
    int no_nuclei = static_cast<int>(global::quantum_sigmas.size());
    std::vector<Eigen::Vector3d> lj_repulsion_gradient(no_nuclei , Eigen::Vector3d::Zero());
    #pragma omp parallel
    {
        std::vector<Eigen::Vector3d> gradient_contr(no_nuclei , Eigen::Vector3d::Zero()) ;
        #pragma omp for
        for(int j = start; j < end; j++){
            for(int i = 0; i < no_nuclei; i++) {
            std::tuple<double, double> combined_sigma_epsilon = (*combination_func)(global::quantum_sigmas[i],
                                                                                    global::classical_sigmas[j],
                                                                                    global::quantum_epsilons[i],
                                                                                    global::classical_epsilons[j]);
            double sigma = std::get<0>(combined_sigma_epsilon);
            double epsilon = std::get<1>(combined_sigma_epsilon);
            double recip_distance = compute_t_tensor((global::nuclei_coordinates[i] - global::coordinates[j]),
                                                      global::tensor_template_potential,
                                                      0, 0, 0, 0)(0,0);
            Eigen::MatrixXd recip_distance_deriv = compute_t_tensor((global::nuclei_coordinates[i] - global::coordinates[j]),
                                                               global::tensor_template_potential,
                                                               1, 0, 1, 0);
            gradient_contr[i] += epsilon * 12.0 * std::pow(sigma, 12) * std::pow(recip_distance, 11) * recip_distance_deriv;
            }
        }
        #pragma omp critical
        {
        for (int i = 0; i < no_nuclei; ++i) {
            lj_repulsion_gradient[i] += 4.0 * gradient_contr[i];
        }
        }
    }
    return lj_repulsion_gradient;
}

// Computes the LJ dispersion gradients of the Nuclei in a QuantumSubsystem interacting with a ClassicalSubsystem.
// Parallelized with OpenMP.
std::vector<Eigen::Vector3d> compute_lj_dispersion_gradient(int start, int end, std::string combination_rule) {
    // Define the function pointer type
    std::tuple<double, double> (*combination_func)(double, double, double, double);
    if (combination_rule == "Lorentz-Berthelot") {
        combination_func = LB_combination;
    }
    int no_nuclei = static_cast<int>(global::quantum_sigmas.size());
    std::vector<Eigen::Vector3d> lj_dispersion_gradient(no_nuclei , Eigen::Vector3d::Zero());
    #pragma omp parallel
    {
        std::vector<Eigen::Vector3d> gradient_contr(no_nuclei , Eigen::Vector3d::Zero()) ;
        #pragma omp for
        for(int j = start; j < end; j++){
            for(int i = 0; i < no_nuclei; i++) {
            std::tuple<double, double> combined_sigma_epsilon = (*combination_func)(global::quantum_sigmas[i],
                                                                                    global::classical_sigmas[j],
                                                                                    global::quantum_epsilons[i],
                                                                                    global::classical_epsilons[j]);
            double sigma = std::get<0>(combined_sigma_epsilon);
            double epsilon = std::get<1>(combined_sigma_epsilon);
            double recip_distance = compute_t_tensor((global::nuclei_coordinates[i] - global::coordinates[j]),
                                                      global::tensor_template_potential,
                                                      0, 0, 0, 0)(0,0);
            Eigen::MatrixXd recip_distance_deriv = compute_t_tensor((global::nuclei_coordinates[i] - global::coordinates[j]),
                                                               global::tensor_template_potential,
                                                               1, 0, 1, 0);
            gradient_contr[i] += -1.0 * epsilon * 6.0 * std::pow(sigma, 6) * std::pow(recip_distance, 5) * recip_distance_deriv;
            }
        }
        #pragma omp critical
        {
        for (int i = 0; i < no_nuclei; ++i) {
            lj_dispersion_gradient[i] += 4.0 * gradient_contr[i];
        }
        }
    }
    return lj_dispersion_gradient;
}

// Computes the derivative of the LJ repulsion potential between a ClassicalSubsystem and a nucleus.
// Parallelized with OpenMP.
double compute_perturbed_lj_repulsion(int start,
                                      int end,
                                      int nuc_idx,
                                      std::string combination_rule,
                                      std::vector<std::vector<std::vector<Eigen::Matrix<int, 2, 3>>>> k_partitions) {
    double perturbed_lj_repulsion = 0.0;
    std::tuple<double, double> (*combination_func)(double, double, double, double);
    if (combination_rule == "Lorentz-Berthelot") {
        combination_func = LB_combination;
    }
    #pragma omp parallel for reduction(+:perturbed_lj_repulsion)
    for (int j = start; j < end; j++) {
        for (const auto& partition_term : k_partitions){
            double partition_contr = 0.0;
            std::size_t length = partition_term.size();
            if (length > 12) {
                partition_contr += 0.0;
            } else {
                double partition_tmp;
                std::tuple<double, double> combined_sigma_epsilon = (*combination_func)(global::quantum_sigmas[nuc_idx],
                                                                                        global::classical_sigmas[j],
                                                                                        global::quantum_epsilons[nuc_idx],
                                                                                        global::classical_epsilons[j]);
                double sigma = std::get<0>(combined_sigma_epsilon);
                double epsilon = std::get<1>(combined_sigma_epsilon);
                double recip_distance = compute_t_tensor((global::nuclei_coordinates[nuc_idx] - global::coordinates[j]),
                                                         global::tensor_template_potential,
                                                         0, 0, 0, 0)(0,0);
                double factorial_prefactor = global::factorials[12] / global::factorials[12 - length];
                partition_tmp = epsilon * factorial_prefactor * std::pow(sigma, 12) * std::pow(recip_distance, 12 - length);
                Eigen::Matrix<int, 2, 3> sum_matrix = Eigen::Matrix<int, 2, 3>::Zero();
                for (const auto& partition : partition_term) {
                    sum_matrix.setZero();
                    for (const auto& element : partition){
                        sum_matrix += element;
                    }
                    partition_tmp *= compute_interaction_tensor_element(sum_matrix,
                                                                         (global::nuclei_coordinates[nuc_idx] - global::coordinates[j]),
                                                                         global::tensor_coefficients);
                }
                partition_contr += partition_tmp;
            }
            perturbed_lj_repulsion += partition_contr;
        }
    }
    return 4.0 * perturbed_lj_repulsion;
}

// Computes the derivative of the LJ dispersion potential between a ClassicalSubsystem and a nucleus.
// Parallelized with OpenMP.
double compute_perturbed_lj_dispersion(int start,
                                        int end,
                                        int nuc_idx,
                                        std::string combination_rule,
                                        std::vector<std::vector<std::vector<Eigen::Matrix<int, 2, 3>>>> k_partitions) {
    double perturbed_lj_dispersion = 0.0;
    std::tuple<double, double> (*combination_func)(double, double, double, double);
    if (combination_rule == "Lorentz-Berthelot") {
        combination_func = LB_combination;
    }
    #pragma omp parallel for reduction(+:perturbed_lj_dispersion)
    for (int j = start; j < end; j++) {
        for (const auto& partition_term : k_partitions){
            double partition_contr = 0.0;
            std::size_t length = partition_term.size();
            if (length > 6) {
                partition_contr += 0.0;
            } else {
                double partition_tmp;
                std::tuple<double, double> combined_sigma_epsilon = (*combination_func)(global::quantum_sigmas[nuc_idx],
                                                                                        global::classical_sigmas[j],
                                                                                        global::quantum_epsilons[nuc_idx],
                                                                                        global::classical_epsilons[j]);
                double sigma = std::get<0>(combined_sigma_epsilon);
                double epsilon = std::get<1>(combined_sigma_epsilon);
                double recip_distance = compute_t_tensor((global::nuclei_coordinates[nuc_idx] - global::coordinates[j]),
                                                         global::tensor_template_potential,
                                                         0, 0, 0, 0)(0,0);
                double factorial_prefactor = global::factorials[6] / global::factorials[6 - length];
                partition_tmp = epsilon * factorial_prefactor * std::pow(sigma, 6) * std::pow(recip_distance, 6 - length);
                Eigen::Matrix<int, 2, 3> sum_matrix = Eigen::Matrix<int, 2, 3>::Zero();
                for (const auto& partition : partition_term) {
                    sum_matrix.setZero();
                    for (const auto& element : partition){
                        sum_matrix += element;
                    }
                    partition_tmp *= compute_interaction_tensor_element(sum_matrix,
                                                                         (global::nuclei_coordinates[nuc_idx] - global::coordinates[j]),
                                                                         global::tensor_coefficients);
                }
                partition_contr += partition_tmp;
            }
            perturbed_lj_dispersion += partition_contr;
        }
    }
    return -4.0 * perturbed_lj_dispersion;
}

}
