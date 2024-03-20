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

// Computes the field caused by induced dipoles at atom i.
// Parallelized with OpenMP.
Eigen::MatrixXd ind_dipoles_field(int i) {
    Eigen::MatrixXd ind_dipoles_field = Eigen::MatrixXd::Zero(3, 1);
    #pragma omp parallel
    {
        Eigen::MatrixXd field_part = Eigen::MatrixXd::Zero(3, 1);
        #pragma omp for
        for(long j = 0; j < (long)global::coordinates.size(); j++) {
            if(global::exclusions[i].find(global::indices[j]) != global::exclusions[i].end()) {
                continue;
            }
            Eigen::Vector3d r_ab = global::coordinates[i] - global::coordinates[j];
            Eigen::MatrixXd t_tensor = compute_t_tensor(r_ab,
                                                        global::tensor_template_potential,
                                                        1, 1, 1, 1);
            field_part += t_tensor * global::old_ind_dipoles.row(j).transpose();
        }
        #pragma omp critical
        {
            ind_dipoles_field += field_part;
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

// Computes the self energy of a ClassicalSubsystem for given array of indexes.
// Parallelized with OpenMP.
double self_energy(Eigen::MatrixXi idx_arr) {
    double self_energy = 0.0;
    #pragma omp parallel reduction(+:self_energy)
    {
        double energy_contr = 0.0;
        #pragma omp for
        for(int k = 0; k < idx_arr.rows(); k++){
            int i = idx_arr(k, 0);
            int j = idx_arr(k, 1);
            if(global::exclusions[i].find(global::indices[j]) != global::exclusions[i].end()) {
                continue;
            }
            Eigen::Vector3d r_ab = global::coordinates[i] - global::coordinates[j];
            Eigen::MatrixXd t_tensor = compute_t_tensor(r_ab,
                                                        global::tensor_template_interaction,
                                                        global::multipole_orders[j], global::multipole_orders[i], 0, 0);
            energy_contr += global::multipoles[j].transpose() * t_tensor * global::multipoles[i];
        }
        self_energy += energy_contr;
    }
    return self_energy;
}
}