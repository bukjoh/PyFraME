import veloxchem as vlx
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def integrals():
    h2o_xyz = """3
    water
    O        0.0000000000      0.0000000000      0.0000000000                 
    H        0.6891400000      0.8324710000      0.0000000000                 
    H        0.7224340000     -0.8726890000      0.0000000000
    """

    act_xyz = """10
    atc
    C                    30.101                    29.705                    29.43
    C                    30.889                    29.91                     30.735
    C                    28.635                    30.016                    29.419
    O                    30.67                     29.421                    28.396
    H                    31.182                    30.941                    30.734
    H                    30.307                    29.618                    31.604
    H                    31.868                    29.391                    30.755
    H                    28.215                    30.575                    30.327
    H                    28.132                    29.059                    29.463
    H                    28.339                    30.503                    28.446
    """
    act_moleule = vlx.Molecule.read_xyz_string(act_xyz)
    act_basis = vlx.MolecularBasis.read(act_moleule, "sto-3g")
    scf_drv = vlx.ScfRestrictedDriver()
    scf_act_results = scf_drv.compute(act_moleule, act_basis)
    #print('HERE0000', repr(2 * scf_act_results['D_alpha']))




    molecule = vlx.Molecule.read_xyz_string(h2o_xyz)
    basis = vlx.MolecularBasis.read(molecule, "sto-3g")
    scf_results = scf_drv.compute(molecule, basis)
    #print(scf_results)

    #electronic part
    mm_sites = [[1.0, 0.0, 0.0]]
    mm_charges = [1.0]

    pot_drv = vlx.NuclearPotentialIntegralsDriver()
    v_es = -1.0 * pot_drv.compute(molecule, basis, mm_charges, mm_sites).to_numpy()
    e_es = np.einsum('ij, ij', scf_results['D_alpha'], v_es)
    #print(e_es)
    ef_drv = vlx.veloxchemlib.ElectricFieldIntegralsDriver()
    ef_results = ef_drv.compute(molecule, basis, dipoles=np.array([[1.0, 1.0, 1.0]]),
                                coordinates=np.array([[1.0, 0.0, 0.0]]))
    #print("ef drv results!", ef_results ,ef_results.x_to_numpy() +  ef_results.y_to_numpy() + ef_results.z_to_numpy())

    #print(v_es)
    #print(e_es)

integrals()
