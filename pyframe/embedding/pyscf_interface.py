import numpy as np

from pyscf import gto
from pyscf import df

import os

from pyframe.embedding import read_input, induction_interactions, electrostatic_interactions

mol = gto.M(atom='''                                                                                                            
        6        0.000000    0.000000   -0.542500          
        8        0.000000    0.000000    0.677500             
        1        0.000000    0.935307   -1.082500
        1        0.000000   -0.935307   -1.082500
            ''', basis='sto3g', verbose=7,
            output='/dev/null')

result = read_input.reader(input_data='/home/jvest/test_pyscf/pyscf_test.json')
core = result[0]
env = result[1]


class PolarizableEmbedding:
    # create molecule from quantum_subsystem?
    def __init__(self, molecule, quantum_subsystem=None, classical_subsystem=None):
        self.mol = molecule
        self.max_memory = mol.max_memory
        self.quantum_subsystem = quantum_subsystem
        self.classical_subsystem = classical_subsystem
        self.f_el_es = self.compute_multipole_potential()
        self.e_nuc_es = electrostatic_interactions.compute_electrostatic_nuclear_energy(
            quantum_subsystem=self.quantum_subsystem,
            classical_subsystem=self.classical_subsystem)

    def compute_multipole_potential(self):
        if np.any(self.classical_subsystem.multipole_orders > 2):
            raise NotImplementedError("""Multipole potential integrals not
                                      implemented for order > 2.""")
        moments = self.classical_subsystem.degenerate_multipoles_with_taylor_coefficients

        op = 0
        # op -= because sign is different compared to vlx?
        # 0 order
        fakemol = gto.fakemol_for_charges(self.classical_subsystem.coordinates)
        integral0 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e')
        moments_0 = np.array([m[0:1] for m in moments])
        op -= np.einsum('ijg,ga->ij', integral0, moments_0)
        # 1 order
        if np.any(self.classical_subsystem.multipole_orders >= 1):
            idx = np.where(self.classical_subsystem.multipole_orders >= 1)[0]
            fakemol = gto.fakemol_for_charges(self.classical_subsystem.coordinates[idx])
            integral1 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ip1')

            moments_1 = np.array([moments[i][1:4] for i in idx])
            v = np.einsum('aijg,ga,a->ij', integral1, moments_1, )
            op -= v + v.T
        if np.any(self.classical_subsystem.multipole_orders >= 2):
            idx = np.where(self.classical_subsystem.multipole_orders >= 2)[0]
            fakemol = gto.fakemol_for_charges(self.classical_subsystem.coordinates[idx])
            n_sites = idx.size
            moments_2_non_symmetrized = np.array([moments[i][4:10] for i in idx])
            moments_2 = np.zeros((n_sites, 9))
            moments_2[:, [0, 1, 2, 4, 5, 8]] = moments_2_non_symmetrized
            moments_2[:, [0, 3, 6, 4, 7, 8]] += moments_2_non_symmetrized
            moments_2 *= .5

            integral2 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ipip1')
            v = np.einsum('aijg,ga->ij', integral2, moments_2)
            op -= v + v.T
            integral2 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ipvip1')
            op -= np.einsum('aijg,ga->ij', integral2, moments_2) * 2
        return op

    def compute_pe_contributions(self, density_matrix):
        density_matrices = np.asarray(density_matrix)
        is_single_dm = density_matrices.ndim == 2

        nao = density_matrices.shape[-1]
        density_matrices = density_matrices.reshape(-1, nao, nao)
        n_dm = density_matrices.shape[0]
        max_memory = self.max_memory

        # very conservative estimate (based on multipole potential integrals)
        # when all sites have a charge, dipole, and quadrupole moment

        e_el_es = np.einsum('ij,xij->x', self.f_el_es, density_matrices)[0]

        ref_f_el_es = np.array([[1.90710369e-04, 4.73652831e-05, 1.78382156e-07, 4.05445876e-07,
                                 1.38721902e-07, 1.96461281e-10, 6.76049672e-06, 6.95473099e-09,
                                 1.58461717e-08, -1.13651367e-05, 1.21899221e-05, 1.21083311e-05],
                                [4.73652831e-05, 1.90710369e-04, 2.08219152e-06, 4.73262564e-06,
                                 1.61925147e-06, 7.04878744e-06, 6.94892438e-05, 6.23020061e-07,
                                 1.43788392e-06, -6.13157984e-05, 9.67494372e-05, 9.16446629e-05],
                                [1.78382156e-07, 2.08219152e-06, 1.90695149e-04, 8.96339631e-08,
                                 2.60264878e-08, 4.00346132e-09, 5.29765185e-07, 4.00547169e-05,
                                 2.00605943e-08, -4.92602106e-07, 9.25886269e-07, 8.37448389e-07],
                                [4.05445876e-07, 4.73262564e-06, 8.96339631e-08, 1.90765074e-04,
                                 8.48425636e-08, 9.30133408e-09, 1.21807419e-06, 2.00605943e-08,
                                 4.00708664e-05, -1.12578404e-06, 8.14042512e-05, -7.33340320e-05],
                                [1.38721902e-07, 1.61925147e-06, 2.60264878e-08, 8.48425636e-08,
                                 1.90670885e-04, 1.18468677e-05, 8.51293647e-05, 7.25581472e-07,
                                 1.67990323e-06, -6.03894313e-05, -4.50864709e-05, -4.28150058e-05],
                                [1.96461281e-10, 7.04878744e-06, 4.00346132e-09, 9.30133408e-09,
                                 1.18468677e-05, 1.94938568e-04, 4.61427265e-05, 1.31670090e-07,
                                 3.06330999e-07, 8.78823387e-08, 9.88004513e-07, 9.85866739e-07],
                                [6.76049672e-06, 6.94892438e-05, 5.29765185e-07, 1.21807419e-06,
                                 8.51293647e-05, 4.61427265e-05, 1.94938568e-04, 1.66213977e-06,
                                 3.86697492e-06, 1.10938430e-06, 1.51229425e-05, 1.45899728e-05],
                                [6.95473099e-09, 6.23020061e-07, 4.00547169e-05, 2.00605943e-08,
                                 7.25581472e-07, 1.31670090e-07, 1.66213977e-06, 1.94929130e-04,
                                 5.76490677e-08, 1.29770195e-08, 1.40419962e-07, 1.30652106e-07],
                                [1.58461717e-08, 1.43788392e-06, 2.00605943e-08, 4.00708664e-05,
                                 1.67990323e-06, 3.06330999e-07, 3.86697492e-06, 5.76490677e-08,
                                 1.94976620e-04, 5.04703122e-08, 8.16875351e-06, -7.26071496e-06],
                                [-1.13651367e-05, -6.13157984e-05, -4.92602106e-07, -1.12578404e-06,
                                 -6.03894313e-05, 8.78823387e-08, 1.10938430e-06, 1.29770195e-08,
                                 5.04703122e-08, 1.94909954e-04, -1.46774821e-05, -1.41519063e-05],
                                [1.21899221e-05, 9.67494372e-05, 9.25886269e-07, 8.14042512e-05,
                                 -4.50864709e-05, 9.88004513e-07, 1.51229425e-05, 1.40419962e-07,
                                 8.16875351e-06, -1.46774821e-05, 1.98524821e-04, 2.77200006e-05],
                                [1.21083311e-05, 9.16446629e-05, 8.37448389e-07, -7.33340320e-05,
                                 -4.28150058e-05, 9.85866739e-07, 1.45899728e-05, 1.30652106e-07,
                                 -7.26071496e-06, -1.41519063e-05, 2.77200006e-05, 1.79187127e-04]])

        fakemol = gto.fakemol_for_charges(self.classical_subsystem.coordinates)
        # first order derivative of the electronic potential integral
        j3c = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ip1')
        electric_fields = (np.einsum('aijg,ij->ga', j3c, density_matrices[0]) +
                           np.einsum('aijg,ji->ga', j3c, density_matrices[0]))
        nuclear_fields = self.quantum_subsystem.compute_nuclear_fields(self.classical_subsystem.coordinates)
        self.classical_subsystem.solve_induced_dipoles(external_fields= (-electric_fields + nuclear_fields))
        # TODO nuclear fields wrong sign????
        #print(self.classical_subsystem.induced_dipoles)
        e_ind = induction_interactions.compute_induction_energy(
            induced_dipoles=self.classical_subsystem.induced_dipoles.
            induced_dipoles,
            total_fields= -electric_fields + nuclear_fields +
                         self.classical_subsystem.multipole_fields)
        f_el_ind = np.einsum('aijg,ga->ij', j3c, self.classical_subsystem.induced_dipoles.
                             induced_dipoles)
        f_el_ind = f_el_ind + f_el_ind.T
        #print(f_el_ind)
        return e_ind + self.e_nuc_es + e_el_es, self.f_el_es - f_el_ind


instance = PolarizableEmbedding(molecule=mol, quantum_subsystem=core, classical_subsystem=env)

input_density = np.array([
    [2.12280187e+00, -4.93652920e-01, 0.00000000e+00, -2.35983379e-18, -5.58513371e-03, 1.53845837e-03, -9.03007827e-03,
     0.00000000e+00, -6.77626358e-20, 1.18757148e-03, -1.21543243e-02, -1.21543243e-02],
    [-4.93652920e-01, 2.03741294e+00, 0.00000000e+00, 9.02598309e-18, 3.64895492e-02, -9.13472107e-03, 2.09723567e-02,
     0.00000000e+00, 1.24683250e-18, -5.14592408e-02, 4.38359626e-02, 4.38359626e-02],
    [0.00000000e+00, 0.00000000e+00, 6.52126455e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
     4.50073255e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
    [-2.35983379e-18, 9.02598309e-18, 0.00000000e+00, 5.71033336e-01, -3.58125530e-18, -3.79470760e-19, 1.50433051e-18,
     0.00000000e+00, 3.54615665e-02, -8.11457563e-19, 2.55307834e-02, -2.55307834e-02],
    [-5.58513371e-03, 3.64895492e-02, 0.00000000e+00, -3.58125530e-18, 6.11110709e-01, -4.83765994e-03, 1.37696114e-02,
     0.00000000e+00, -3.04931861e-19, 4.41505593e-03, -1.10191499e-02, -1.10191499e-02],
    [1.53845837e-03, -9.13472107e-03, 0.00000000e+00, -3.79470760e-19, -4.83765994e-03, 2.11511132e+00, -4.89815305e-01,
     0.00000000e+00, 7.62329653e-20, -1.50658112e-02, -2.53355896e-03, -2.53355896e-03],
    [-9.03007827e-03, 2.09723567e-02, 0.00000000e+00, 1.50433051e-18, 1.37696114e-02, -4.89815305e-01, 2.10157970e+00,
     0.00000000e+00, -3.38813179e-19, 6.15155822e-02, 1.19874343e-02, 1.19874343e-02],
    [0.00000000e+00, 0.00000000e+00, 4.50073255e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
     1.25868374e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
    [-6.77626358e-20, 1.24683250e-18, 0.00000000e+00, 3.54615665e-02, -3.04931861e-19, 7.62329653e-20, -3.38813179e-19,
     0.00000000e+00, 1.26401517e+00, 3.38813179e-19, 1.67996930e-02, -1.67996930e-02],
    [1.18757148e-03, -5.14592408e-02, 0.00000000e+00, -8.11457563e-19, 4.41505593e-03, -1.50658112e-02, 6.15155822e-02,
     0.00000000e+00, 3.38813179e-19, 1.19905448e+00, -1.24242430e-02, -1.24242430e-02],
    [-1.21543243e-02, 4.38359626e-02, 0.00000000e+00, 2.55307834e-02, -1.10191499e-02, -2.53355896e-03, 1.19874343e-02,
     0.00000000e+00, 1.67996930e-02, -1.24242430e-02, 9.45707426e-01, 1.79597856e-02],
    [-1.21543243e-02, 4.38359626e-02, 0.00000000e+00, -2.55307834e-02, -1.10191499e-02, -2.53355896e-03, 1.19874343e-02,
     0.00000000e+00, -1.67996930e-02, -1.24242430e-02, 1.79597856e-02, 9.45707426e-01]
])
ref_energy = -8.551984169348457e-05

# input_density = input_density + input_density.T


# liste von density matrices?
# instance.compute_pe_contributions(density_matrix=input_density)
print(instance.compute_pe_contributions(density_matrix=input_density))
