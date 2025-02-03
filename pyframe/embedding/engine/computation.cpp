#include "computation.h"
#include "fmm/tree.hh"
#include <iostream>


namespace computation
{

// Uses the Lorentz-Berthelot combination rules for non-bonded VdW interactions.
std::tuple<double, double> LB_combination(double sigma_i, double sigma_j, double epsilon_i, double epsilon_j) {
    double comb_sigma = 0.5 * (sigma_i + sigma_j);
    double comb_epsilon = std::sqrt(epsilon_i * epsilon_j);
    return std::make_tuple(comb_sigma, comb_epsilon);
}

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
    // element *= std::pow(-1, multiindex.row(1).sum());
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
                tensor_template(i, j), r_ab, global::tensor_coefficients) * std::pow(-1, tensor_template(i, j).row(1).sum());
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
                (tensor_template(i, j) + perturbation_tuple), r_ab, global::tensor_coefficients) * std::pow(-1, (tensor_template(i, j) + perturbation_tuple).row(1).sum());
            interaction_tensor(i - start_a, j - start_b) = interaction_element;
        }
    }
    return interaction_tensor;
}


Eigen::Vector3d atom_dist(int i, int j) {
    Eigen::Vector3d r_ab = global::coordinates[i] - global::coordinates[j];
    return r_ab;
}

Eigen::Vector3d atom_dist_mic(int i, int j) {
    Eigen::Vector3d r_ij = global::coordinates_scaled[j] - global::coordinates_scaled[i];
    Eigen::Vector3d r_ab = global::box * (r_ij - r_ij.unaryExpr([](double val) { return std::round(val); }));
    return r_ab;
}


// Computes the field caused by induced dipoles at atom i.
// Parallelized with OpenMP.
Eigen::MatrixXd ind_dipoles_field(int start, int end, bool mic) {
    int no_atoms = static_cast<int>(global::coordinates.size());
    Eigen::MatrixXd ind_dipoles_field = Eigen::MatrixXd::Zero(no_atoms, 3);
    Eigen::Vector3d (*dist_func)(int, int);
    if (mic) {
        dist_func = atom_dist_mic;
    } else {
        dist_func = atom_dist;
    }
    for(int i = start; i < end; i++) {
        #pragma omp parallel
        {
            #pragma omp for
            for(int j = 0; j < no_atoms; j++) {
                if(global::exclusions[i].find(global::indices[j]) != global::exclusions[i].end()) {
                    continue;
                }

                Eigen::Vector3d r_ab = dist_func(i, j);
                Eigen::MatrixXd t_tensor = compute_t_tensor(r_ab,
                                                            global::tensor_template_potential,
                                                            1, 1, 1, 1);
                #pragma omp critical
                ind_dipoles_field.row(i) += t_tensor * global::old_ind_dipoles.col(j);
            }
        }
    }
    return ind_dipoles_field;
}

Eigen::MatrixXd ind_dipoles_field_fmm(int n_crit, int order, double theta, double damping) {
    // Validate global::coordinates
    int nparticles = static_cast<int>(global::coordinates.size());
    if (nparticles == 0) {
        throw std::runtime_error("No particles found in global::coordinates.");
    }

    std::vector<double> S(3 * nparticles);
    for (int i = 0; i < nparticles; ++i) {
        S[i * 3 + 0] = global::old_ind_dipoles.col(i)(0);  // x-component global::old_ind_dipoles.row(i)
        S[i * 3 + 1] = global::old_ind_dipoles.col(i)(1);  // y-component -> change back to .row
        S[i * 3 + 2] = global::old_ind_dipoles.col(i)(2);  // z-component
    }
    // Allocate storage for induced fields
    std::vector<double> induced_fields_v(3 * nparticles);

    // Build the FMM tree
    std::shared_ptr<Tree<1, 3>> tree =
        build_shared_tree<1, 3>(S.data() ,n_crit, order, theta, damping);

    // Compute fields using FMM
    tree->compute_field_fmm(induced_fields_v.data());

    // Map the vector correctly with row-major storage
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> ind_dipoles_field =
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(induced_fields_v.data(), nparticles, 3);
    return ind_dipoles_field;
}

Eigen::MatrixXd multipole_fields_fmm(int n_crit, int order, double theta, double damping) {

    int nparticles = static_cast<int>(global::coordinates.size());
    std::vector<double> charges(nparticles, 0.0);
    std::vector<double> dipoles(3 * nparticles, 0.0);
    std::vector<double> quadrupoles(6 * nparticles, 0.0);
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> multipole_fields =
        Eigen::MatrixXd::Zero(nparticles, 3);
    int max_order = global::multipole_orders.maxCoeff();
    // Validate global::coordinates
    if (nparticles == 0) {
        throw std::runtime_error("No particles found in global::coordinates.");
    }
    // Validate max_order not > 2
    if (max_order > 2) {
      throw std::runtime_error(
            "multipole_fields_fmm only support up to quadrupoles (second order).");
    }

    for (int i = 0; i < nparticles; ++i) {
        if (global::multipole_orders[i] >= 0 ) {
            charges[i] = global::multipoles[i].coeff(0);
        }
        if (global::multipole_orders[i] >= 1 ) {
            dipoles[i * 3 + 0] = global::multipoles[i].coeff(1);  // x-component
            dipoles[i * 3 + 1] = global::multipoles[i].coeff(2);  // y-component
            dipoles[i * 3 + 2] = global::multipoles[i].coeff(3);  // z-component
        }
        if (global::multipole_orders[i] >= 2 ) {
          quadrupoles[i * 6 + 0] = global::multipoles[i].coeff(4);
          quadrupoles[i * 6 + 1] = global::multipoles[i].coeff(5);
          quadrupoles[i * 6 + 2] = global::multipoles[i].coeff(6);
          quadrupoles[i * 6 + 3] = global::multipoles[i].coeff(7);
          quadrupoles[i * 6 + 4] = global::multipoles[i].coeff(8);
          quadrupoles[i * 6 + 5] = global::multipoles[i].coeff(9);
        }
    }
    // field contributions from charges
    std::vector<double> fields_v0(3 * nparticles);
    std::shared_ptr<Tree<0, 3>> tree_c = build_shared_tree<0, 3>(
    charges.data(), n_crit, order, theta, damping);
    tree_c->compute_field_fmm(fields_v0.data());
    multipole_fields += Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(fields_v0.data(), nparticles, 3);

    // field contributions from dipoles
    if (max_order > 0) {
    std::vector<double> fields_v1(3 * nparticles);
    std::shared_ptr<Tree<1, 3>> tree_d = build_shared_tree<1, 3>(
    dipoles.data(), n_crit, order, theta, damping);
    tree_d->compute_field_fmm(fields_v1.data());
    multipole_fields += Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(fields_v1.data(), nparticles, 3);
    }
    // field contributions from quadrupoles
    if (max_order > 1) {
    std::vector<double> fields_v2(3 * nparticles);
    std::shared_ptr<Tree<2, 3>> tree_q = build_shared_tree<2, 3>(
    quadrupoles.data(), n_crit, order, theta, damping);
    tree_q->compute_field_fmm(fields_v2.data());
    multipole_fields += Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(fields_v2.data(), nparticles, 3);
    }
    return multipole_fields;
}

// Computes the field caused by induced dipoles at atom i.
// Parallelized with OpenMP.
Eigen::MatrixXd target_source_ind_dipoles_field(Eigen::VectorXi targets, Eigen::VectorXi sources, bool mic) {
    Eigen::MatrixXd ind_dipoles_field = Eigen::MatrixXd::Zero(targets.size(), 3);
    Eigen::Vector3d (*dist_func)(int, int);
    if (mic) {
        dist_func = atom_dist_mic;
    } else {
        dist_func = atom_dist;
    }
    for (int i = 0; i < targets.size(); i++) {
        #pragma omp parallel
        {
            #pragma omp for
            for(int j = 0; j < sources.size(); j++) {
                if(global::exclusions[targets[i]].find(global::indices[sources[j]]) != global::exclusions[targets[i]].end()) {
                    continue;
                }
                Eigen::Vector3d r_ab = dist_func(targets[i], sources[j]);
                Eigen::MatrixXd t_tensor = compute_t_tensor(r_ab,
                                                            global::tensor_template_potential,
                                                            1, 1, 1, 1);
                #pragma omp critical
                ind_dipoles_field.row(i) += (t_tensor * global::old_ind_dipoles.col(sources[j]));
            }
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

// TODO implement MIC here
// Computes the field caused by all atoms at atom i.
// Parallelized with OpenMP.
Eigen::MatrixXd multipole_field(int i) {
    Eigen::MatrixXd multipole_field = Eigen::MatrixXd::Zero(3, 1);
    int no_atoms = static_cast<int>(global::coordinates.size());
    #pragma omp parallel
    {
        Eigen::MatrixXd field_part = Eigen::MatrixXd::Zero(3, 1);
        #pragma omp for
        for(int j = 0; j < no_atoms; j++) {
            if(global::exclusions[i].find(global::indices[j]) != global::exclusions[i].end()) {
                continue;
            }
            Eigen::Vector3d r_ab = global::coordinates[i] - global::coordinates[j];
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

// Computes electrostatic interaction energy for given array of indexes
// Parallelized with OpenMP.
double electrostatic_environment_energy(Eigen::MatrixXi idx_arr) {
    double environment_energy = 0.0;
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
            Eigen::Vector3d r_ab = global::coordinates[i] - global::coordinates[j];
            Eigen::MatrixXd t_tensor = compute_t_tensor(r_ab,
                                                        global::tensor_template_interaction,
                                                        global::multipole_orders[j], global::multipole_orders[i], 0, 0);
            energy_contr += global::multipoles[j].transpose() * t_tensor * global::multipoles[i];
        }
    #pragma omp critical
    {
    environment_energy += energy_contr;
    }
    }
    return environment_energy;
}

// Computes nonelectrostatic interaction energy using an LJ 12-6 potential for given array of indexes
// Parallelized with OpenMP.
double lj_repulsion_environment_energy(Eigen::MatrixXi idx_arr, std::string combination_rule){
    double environment_energy = 0.0;
    std::tuple<double, double> (*combination_func)(double, double, double, double);
    if (combination_rule == "Lorentz-Berthelot") {
        combination_func = LB_combination;
    }
    if (combination_rule != "Lorentz-Berthelot") {
    throw std::invalid_argument("Unsupported combination rule: " + combination_rule);
    }
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
            std::tuple<double, double> combined_sigma_epsilon = (*combination_func)(global::classical_sigmas[i],
                                                                                    global::classical_sigmas[j],
                                                                                    global::classical_epsilons[i],
                                                                                    global::classical_epsilons[j]);
            double sigma = std::get<0>(combined_sigma_epsilon);
            double epsilon = std::get<1>(combined_sigma_epsilon);
            double recip_distance = compute_t_tensor((global::coordinates[i] - global::coordinates[j]),
                                                     global::tensor_template_potential,
                                                     0, 0, 0, 0)(0,0);
            energy_contr += epsilon * std::pow(sigma, 12) * std::pow(recip_distance, 12);
        }
    #pragma omp critical
    {
    environment_energy += energy_contr;
    }
    }
    return 4.0 * environment_energy;
}

// Computes dispersion interaction energy using an LJ 12-6 potential for given array of indexes
// Parallelized with OpenMP.
double lj_dispersion_environment_energy(Eigen::MatrixXi idx_arr, std::string combination_rule){
    double environment_energy = 0.0;
    std::tuple<double, double> (*combination_func)(double, double, double, double);
    if (combination_rule == "Lorentz-Berthelot") {
        combination_func = LB_combination;
    }
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
            std::tuple<double, double> combined_sigma_epsilon = (*combination_func)(global::classical_sigmas[i],
                                                                                    global::classical_sigmas[j],
                                                                                    global::classical_epsilons[i],
                                                                                    global::classical_epsilons[j]);
            double sigma = std::get<0>(combined_sigma_epsilon);
            double epsilon = std::get<1>(combined_sigma_epsilon);
            double recip_distance = compute_t_tensor((global::coordinates[i] - global::coordinates[j]),
                                                     global::tensor_template_potential,
                                                     0, 0, 0, 0)(0,0);
            energy_contr += epsilon * std::pow(sigma, 6) * std::pow(recip_distance, 6);
        }
    #pragma omp critical
    {
    environment_energy += energy_contr;
    }
    }
    return (-1.0) * 4.0 * environment_energy;
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

// Computes the perturbed energy between all atoms and a nucleus.
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

// Computes the energy gradient between all atoms and all nuclei.
// Parallelized with OpenMP.
std::vector<Eigen::Vector3d> e_nuc_es_gradients(int start, int end) {
    int no_nuclei = static_cast<int>(global::nuclei_coordinates.size());
    std::vector<Eigen::Vector3d> e_nuc_es_gradients(no_nuclei , Eigen::Vector3d::Zero());
    #pragma omp parallel
    {
        std::vector<Eigen::Vector3d> thread_gradients(no_nuclei, Eigen::Vector3d::Zero());
        const Eigen::Matrix<int, 2, 3> x_grad((Eigen::Matrix<int, 2, 3>() << 1, 0, 0, 0, 0, 0).finished());
        const Eigen::Matrix<int, 2, 3> y_grad((Eigen::Matrix<int, 2, 3>() << 0, 1, 0, 0, 0, 0).finished());
        const Eigen::Matrix<int, 2, 3> z_grad((Eigen::Matrix<int, 2, 3>() << 0, 0, 1, 0, 0, 0).finished());
        #pragma omp for
        for(int j = start; j < end; j++){
            for(int i = 0; i < no_nuclei; i++) {
                double energy_contr_x = 0.0;
                double energy_contr_y = 0.0;
                double energy_contr_z = 0.0;
                Eigen::Vector3d r_ab = global::nuclei_coordinates[i] - global::coordinates[j];
                Eigen::MatrixXd x_grad_t_tensor = compute_perturbed_t_tensor(r_ab,
                                                          global::tensor_template_interaction,
                                                          x_grad,
                                                          global::multipole_orders[j], 0, 0, 0);
                Eigen::MatrixXd y_grad_t_tensor = compute_perturbed_t_tensor(r_ab,
                                                          global::tensor_template_interaction,
                                                          y_grad,
                                                          global::multipole_orders[j], 0, 0, 0);
                Eigen::MatrixXd z_grad_t_tensor = compute_perturbed_t_tensor(r_ab,
                                                          global::tensor_template_interaction,
                                                          z_grad,
                                                          global::multipole_orders[j], 0, 0, 0);
                energy_contr_x += (global::multipoles[j].transpose() * x_grad_t_tensor * global::nuclei_charges[i])[0];
                energy_contr_y += (global::multipoles[j].transpose() * y_grad_t_tensor * global::nuclei_charges[i])[0];
                energy_contr_z += (global::multipoles[j].transpose() * z_grad_t_tensor * global::nuclei_charges[i])[0];
                thread_gradients[i][0] += energy_contr_x;
                thread_gradients[i][1] += energy_contr_y;
                thread_gradients[i][2] += energy_contr_z;
            }
        }
        #pragma omp critical
        {
            for(int i = 0; i < no_nuclei; ++i) {
                e_nuc_es_gradients[i] += thread_gradients[i];
            }
        }
    }
    return e_nuc_es_gradients;
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
