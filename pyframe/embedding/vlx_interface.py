import veloxchem as vlx
import numpy as np
import os
import scipy
import json

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
                        coordinates: list,
                        density: np.ndarray
                       ):
        # permanent electric field used to calulate induced dipoles by electron density
        electric_fields = np.zeros([len(coordinates), 3])
        electric_field = np.zeros(3)
        for i, coordinate in enumerate(coordinates):
            ef_results = self.ef_drv.compute(self.molecule, self.basis, coordinate[0], coordinate[1], coordinate[2])
            electric_field[0] = np.einsum("ab, ab", density, ef_results.x_to_numpy())
            electric_field[1] = np.einsum("ab, ab", density, ef_results.y_to_numpy())
            electric_field[2] = np.einsum("ab, ab", density, ef_results.z_to_numpy())
            electric_fields[i] = electric_field
        return -1.0 * electric_fields

    def multipole_field_integrals(self,
                                  dipoles: list,
                                  coordinates: list
                                  ):
        # use induced dipoles here to get fock matrix contributions
        ef_results = self.ef_drv.compute(self.molecule, self.basis, dipoles, coordinates).to_numpy()
        return -1.0 * (ef_results.x_to_numpy() + ef_results.y_to_numpy() + ef_results.z_to_numpy())


def scf_solver(h, V_nuc, C, nocc, g, S):
    max_iter = 50
    conv_thresh = 1e-4

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
