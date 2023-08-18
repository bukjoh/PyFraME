import veloxchem as vlx
import numpy as np
import os
import scipy
from pyframe.embedding import induction_interactions
from typing import Optional

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class EmbeddingIntegralDriver:
    """Interface to the PyFraME embedding library.

    Args:
        molecule:
        basis:
    """

    def __init__(self,
                 molecule: str,
                 basis: str
                 ):
        self.molecule = vlx.Molecule.read_xyz_string(xyz=molecule)
        self.basis = vlx.MolecularBasis.read(mol=self.molecule, basis_name=basis)
        self.pot_drv = vlx.NuclearPotentialIntegralsDriver()
        self.ef_drv = vlx.veloxchemlib.ElectricFieldIntegralsDriver()

    def multipole_potential_integrals(self,
                                      charges: list,
                                      coordinates: list
                                      ):
        return -1.0 * self.pot_drv.compute(self.molecule, self.basis, charges, coordinates).to_numpy()

    def electric_fields(self,
                        coordinates: np.ndarray,
                        density: np.ndarray
                        ):
        # permanent electric field used to calculate induced dipoles by electron density
        electric_fields = np.zeros([len(coordinates), 3])
        electric_field = np.zeros(3)
        for i, coordinate in enumerate(coordinates):
            ef_results = self.ef_drv.compute(self.molecule, self.basis, coordinate[0], coordinate[1], coordinate[2])
            electric_field[0] = np.einsum("ij, ij", density, ef_results.x_to_numpy())
            electric_field[1] = np.einsum("ij, ij", density, ef_results.y_to_numpy())
            electric_field[2] = np.einsum("ij, ij", density, ef_results.z_to_numpy())
            electric_fields[i, :] = electric_field
        return -1.0 * electric_fields

    def multipole_field_integrals(self,
                                  dipoles: np.ndarray,
                                  coordinates: np.ndarray
                                  ):
        # use induced dipoles here to get fock matrix contributions
        ef_results = self.ef_drv.compute(self.molecule, self.basis, dipoles, coordinates).to_numpy()
        return -1.0 * (ef_results[0] + ef_results[1] + ef_results[2])


def scf_solver(h, V_nuc, C, nocc, g, S, conv_thresh: Optional[float] = None):
    max_iter = 100
    if conv_thresh is None:
        conv_thresh = 1e-10

    print("iter      SCF energy    Error norm")

    for iter in range(max_iter):

        D_alpha = np.einsum("ik,jk->ij", C[:, :nocc], C[:, :nocc])
        J = np.einsum("ijkl,kl->ij", g, D_alpha)
        K = np.einsum("ilkj,kl->ij", g, D_alpha)
        F = h + 2 * J - K

        E = np.einsum("ij,ij->", h + F, D_alpha) + V_nuc

        # compute convergence metric
        F_MO = np.einsum("ki,kl,lj->ij", C, F, C)
        e_vec = np.reshape(F_MO[:nocc, nocc:], -1)
        error = np.linalg.norm(e_vec)
        print(f"{iter:>2d}  {E:16.8f}  {error:10.2e}")

        if error < conv_thresh:
            print("SCF iterations converged!")
            break

        epsilon, C = scipy.linalg.eigh(F, S)

    return E, C


def scf_pe_solver(h, V_nuc, C, nocc, g, S, embedding_driver, core, env):
    max_iter = 100
    conv_thresh = 1e-10
    E, ind_dipoles, e_ind, electric_fields = None, None, None, None
    print("iter      SCF energy    Error norm")
    static_drv = induction_interactions.compute_static_contributions
    ind_dip_drv = induction_interactions.compute_induced_dipoles
    ind_int_drv = induction_interactions.compute_induction_interaction
    coordinates, multipole_fields, nuclear_fields, polarizabilities, classical_fragments = (
        static_drv(quantum_subsystem=core,
                   classical_subsystem=env))
    static_fields = multipole_fields + nuclear_fields
    for iter in range(max_iter):
        D_alpha = np.einsum("ik,jk->ij", C[:, :nocc], C[:, :nocc])
        core.update_density(2 * D_alpha)
        # take density -> recalculate the induced dipoles -> recalculate fock contributions
        ind_dipoles, electric_fields = ind_dip_drv(density=core.density_matrix.density,
                                                   integral_drv=embedding_driver,
                                                   coordinates=coordinates,
                                                   static_fields=static_fields,
                                                   polarizabilities=polarizabilities,
                                                   classical_fragments=classical_fragments,
                                                   threshold=1e-10)
        total_fields = static_fields + electric_fields
        e_ind, h_ind = ind_int_drv(induced_dipoles=ind_dipoles,
                                   total_fields=total_fields,
                                   coordinates=coordinates,
                                   integral_drv=embedding_driver)
        J = np.einsum("ijkl,kl->ij", g, D_alpha)
        K = np.einsum("ilkj,kl->ij", g, D_alpha)
        F = h + 2 * J - K - h_ind

        E = np.einsum("ij,ij->", h + F + h_ind, D_alpha) + V_nuc + e_ind

        # compute convergence metric
        F_MO = np.einsum("ki,kl,lj->ij", C, F, C)
        e_vec = np.reshape(F_MO[:nocc, nocc:], -1)
        error = np.linalg.norm(e_vec)
        print(f"{iter:>2d}  {E:16.8f}  {error:10.2e}")

        if error < conv_thresh:
            print("SCF iterations converged!")
            break

        epsilon, C = scipy.linalg.eigh(F, S)
    e_nuc_ind = induction_interactions.compute_induction_energy(induced_dipoles=ind_dipoles, fields=nuclear_fields)
    e_mul_ind = induction_interactions.compute_induction_energy(induced_dipoles=ind_dipoles, fields=multipole_fields)
    e_el_ind = induction_interactions.compute_induction_energy(induced_dipoles=ind_dipoles, fields=electric_fields)
    return E, C, ind_dipoles, e_ind, e_nuc_ind, e_mul_ind, e_el_ind

