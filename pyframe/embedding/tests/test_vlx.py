import veloxchem as vlx

def test_integrals():
    h2o_xyz = """3
    water
    O        0.0000000000      0.0000000000      0.0000000000                 
    H        0.6891400000      0.8324710000      0.0000000000                 
    H        0.7224340000     -0.8726890000      0.0000000000
    """
    molecule = vlx.Molecule.read_xyz_string(h2o_xyz)
    basis = vlx.MolecularBasis.read(molecule, "sto-3g")

    #electronic part
    mm_sites = [[1.0, 0.0, 0.0]]
    mm_charges = [1.0]

    pot_drv = vlx.NuclearPotentialIntegralsDriver()
    v_es = -1.0 * pot_drv.compute(molecule, basis, mm_charges, mm_sites).to_numpy()
    print(v_es)