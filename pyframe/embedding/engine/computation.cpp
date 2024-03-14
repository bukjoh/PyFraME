#include "computation.h"

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

// Computes the field caused by induced dipoles at site i.
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
}