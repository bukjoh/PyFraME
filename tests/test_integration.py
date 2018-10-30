# coding=utf-8
"""Integration tests"""

import pytest
import os
import filecmp

import pyframe
import pyframe.readers
from pyframe.readers import PDBError


def test_pdb_error():
    test = 'pdb_error'
    tests_dir = '{0}'.format(os.path.dirname(__file__))
    with pytest.raises(PDBError):
        pyframe.MolecularSystem(input_file='{0}/{1}/{1}.pdb'.format(tests_dir, test))

def test_permanganate():
    test = 'permanganate'
    tests_dir = '{0}'.format(os.path.dirname(__file__))
    project = pyframe.Project(work_dir='{0}'.format(tests_dir))
    system = pyframe.MolecularSystem(input_file='{0}/{1}/{1}.pdb'.format(tests_dir, test), bond_threshold=1.15)
    core = system.get_fragments_by_name('LIG')
    system.set_core_region(core)
    solvent = system.get_fragments_by_name(names=['HOH'])
    system.add_region(name='solvent', fragments=solvent, use_standard_potentials=True, standard_potential_model='TIP3P')
    project.create_embedding_potential(system)
    project.write_core(system)
    assert os.path.isfile('{0}/{1}/{1}.mol'.format(tests_dir, test))
    assert filecmp.cmp('{0}/{1}/{1}.mol'.format(tests_dir, test), '{0}/{1}/{1}.mol.ref'.format(tests_dir, test))
    os.remove('{0}/{1}/{1}.mol'.format(tests_dir, test))
    project.write_potential(system)
    assert os.path.isfile('{0}/{1}/{1}.pot'.format(tests_dir, test))
    assert filecmp.cmp('{0}/{1}/{1}.pot'.format(tests_dir, test), '{0}/{1}/{1}.pot.ref'.format(tests_dir, test))
    os.remove('{0}/{1}/{1}.pot'.format(tests_dir, test))


def test_pna_in_ccl4():
    test = 'PNA_in_CCl4'
    tests_dir = '{0}'.format(os.path.dirname(__file__))
    project = pyframe.Project(work_dir='{0}'.format(tests_dir))
    system = pyframe.MolecularSystem(input_file='{0}/{1}/{1}.pdb'.format(tests_dir, test), bond_threshold=1.15)
    core = system.get_fragments_by_name(names=['PNA'])
    system.set_core_region(core)
    solvent = system.get_fragments_by_name(names=['TET'])
    system.add_region(name='solvent', fragments=solvent, use_standard_potentials=True, standard_potential_model='SEP')
    project.create_embedding_potential(system)
    project.write_core(system)
    assert os.path.isfile('{0}/{1}/{1}.mol'.format(tests_dir, test))
    assert filecmp.cmp('{0}/{1}/{1}.mol'.format(tests_dir, test), '{0}/{1}/{1}.mol.ref'.format(tests_dir, test))
    os.remove('{0}/{1}/{1}.mol'.format(tests_dir, test))
    project.write_potential(system)
    assert os.path.isfile('{0}/{1}/{1}.pot'.format(tests_dir, test))
    assert filecmp.cmp('{0}/{1}/{1}.pot'.format(tests_dir, test), '{0}/{1}/{1}.pot.ref'.format(tests_dir, test))
    os.remove('{0}/{1}/{1}.pot'.format(tests_dir, test))


def test_4np_in_water():
    test = '4NP_in_water'
    tests_dir = '{0}'.format(os.path.dirname(__file__))
    project = pyframe.Project(work_dir='{0}'.format(tests_dir))
    system = pyframe.MolecularSystem(input_file='{0}/{1}/{1}.pdb'.format(tests_dir, test), bond_threshold=1.15)
    core = system.get_fragments_by_name(names=['4NP'])
    system.set_core_region(core)
    ions = system.get_fragments_by_number(numbers=[2, *range(3, 8), 8, 9]) 
    ions += system.get_fragments_by_number(10)
    system.add_region(name='ions', fragments=ions, use_standard_potentials=True, standard_potential_model='SEP')
    tip3p = system.get_fragments_by_number(numbers=[*range(11, 16)])
    system.add_region(name='tip3p', fragments=tip3p, use_standard_potentials=True, standard_potential_model='TIP3P')
    solvent = system.get_fragments_by_name(names=['WAT'])
    system.add_region(name='solvent', fragments=solvent, use_standard_potentials=True, standard_potential_model='SEP')
    project.create_embedding_potential(system)
    project.write_core(system)
    assert os.path.isfile('{0}/{1}/{1}.mol'.format(tests_dir, test))
    assert filecmp.cmp('{0}/{1}/{1}.mol'.format(tests_dir, test), '{0}/{1}/{1}.mol.ref'.format(tests_dir, test))
    os.remove('{0}/{1}/{1}.mol'.format(tests_dir, test))
    project.write_potential(system)
    assert os.path.isfile('{0}/{1}/{1}.pot'.format(tests_dir, test))
    assert filecmp.cmp('{0}/{1}/{1}.pot'.format(tests_dir, test), '{0}/{1}/{1}.pot.ref'.format(tests_dir, test))
    os.remove('{0}/{1}/{1}.pot'.format(tests_dir, test))


def test_insulin():
    test = 'insulin'
    tests_dir = '{0}'.format(os.path.dirname(__file__))
    project = pyframe.Project(work_dir='{0}'.format(tests_dir))
    system = pyframe.MolecularSystem(input_file='{0}/{1}/{1}.pdb'.format(tests_dir, test), bond_threshold=1.15)
    protein = system.get_fragments_by_chain_id(chain_ids=['A', 'B'])
    system.add_region(name='protein', fragments=protein, use_mfcc=True, use_multipoles=True, multipole_order=2,
                      multipole_xcfun='PBE0', multipole_basis='loprop-cc-pVDZ', use_polarizabilities=True,
                      polarizability_xcfun='PBE0', polarizability_basis='loprop-cc-pVDZ',
                      isotropic_polarizabilities=True)
    project.create_embedding_potential(system)
    project.write_potential(system)
    assert os.path.isfile('{0}/{1}/{1}.pot'.format(tests_dir, test))
    potential = pyframe.readers.read_pelib_potential('{0}/{1}/{1}.pot'.format(tests_dir, test))
    reference_potential = pyframe.readers.read_pelib_potential('{0}/{1}/{1}.pot.ref'.format(tests_dir, test))
    for site, ref_site in zip(potential.values(), reference_potential.values()):
        assert site.element == ref_site.element
        for comp, ref_comp in zip(site.coordinate, ref_site.coordinate):
            assert pytest.approx(comp) == ref_comp
        for comp, ref_comp in zip(site.M0, ref_site.M0):
            assert pytest.approx(comp) == ref_comp
        for comp, ref_comp in zip(site.M1, ref_site.M1):
            assert pytest.approx(comp) == ref_comp
        for comp, ref_comp in zip(site.M2, ref_site.M2):
            assert pytest.approx(comp) == ref_comp
        for comp, ref_comp in zip(site.P11, ref_site.P11):
            assert pytest.approx(comp) == ref_comp
    os.remove('{0}/{1}/{1}.pot'.format(tests_dir, test))


def test_insulin_pfp_fragment():
    test = 'insulin_pfp_fragment'
    tests_dir = '{0}'.format(os.path.dirname(__file__))
    project = pyframe.Project(work_dir='{0}'.format(tests_dir))
    system = pyframe.MolecularSystem(input_file='{0}/{1}/{1}.pdb'.format(tests_dir, test), bond_threshold=1.15)
    protein = system.get_fragments_by_chain_id(chain_ids=['A', 'B'])
    system.add_region(name='protein', fragments=protein, use_standard_potentials=True,
                      standard_potential_model='PFP', standard_potential_exclusion_type='fragment')
    project.create_embedding_potential(system)
    project.write_potential(system)
    assert os.path.isfile('{0}/{1}/{1}.pot'.format(tests_dir, test))
    assert filecmp.cmp('{0}/{1}/{1}.pot'.format(tests_dir, test), '{0}/{1}/{1}.pot.ref'.format(tests_dir, test))
    os.remove('{0}/{1}/{1}.pot'.format(tests_dir, test))


def test_insulin_pfp_mfcc():
    test = 'insulin_pfp_mfcc'
    tests_dir = '{0}'.format(os.path.dirname(__file__))
    project = pyframe.Project(work_dir='{0}'.format(tests_dir))
    system = pyframe.MolecularSystem(input_file='{0}/{1}/{1}.pdb'.format(tests_dir, test), bond_threshold=1.15)
    protein = system.get_fragments_by_chain_id(chain_ids=['A', 'B'])
    system.add_region(name='protein', fragments=protein, use_standard_potentials=True,
                      standard_potential_model='PFP', standard_potential_exclusion_type='mfcc')
    project.create_embedding_potential(system)
    project.write_potential(system)
    assert os.path.isfile('{0}/{1}/{1}.pot'.format(tests_dir, test))
    assert filecmp.cmp('{0}/{1}/{1}.pot'.format(tests_dir, test), '{0}/{1}/{1}.pot.ref'.format(tests_dir, test))
    os.remove('{0}/{1}/{1}.pot'.format(tests_dir, test))


def test_insulin_ff94():
    test = 'insulin_ff94'
    tests_dir = '{0}'.format(os.path.dirname(__file__))
    project = pyframe.Project(work_dir='{0}'.format(tests_dir))
    system = pyframe.MolecularSystem(input_file='{0}/{1}/{1}.pdb'.format(tests_dir, test), bond_threshold=1.15)
    protein = system.get_fragments_by_chain_id(chain_ids=['A', 'B'])
    system.add_region(name='protein', fragments=protein, use_standard_potentials=True,
                      standard_potential_model='ff94')
    project.create_embedding_potential(system)
    project.write_potential(system)
    assert os.path.isfile('{0}/{1}/{1}.pot'.format(tests_dir, test))
    assert filecmp.cmp('{0}/{1}/{1}.pot'.format(tests_dir, test), '{0}/{1}/{1}.pot.ref'.format(tests_dir, test))
    os.remove('{0}/{1}/{1}.pot'.format(tests_dir, test))


def test_insulin_ff03():
    test = 'insulin_ff03'
    tests_dir = '{0}'.format(os.path.dirname(__file__))
    project = pyframe.Project(work_dir='{0}'.format(tests_dir))
    system = pyframe.MolecularSystem(input_file='{0}/{1}/{1}.pdb'.format(tests_dir, test), bond_threshold=1.15)
    protein = system.get_fragments_by_chain_id(chain_ids=['A', 'B'])
    system.add_region(name='protein', fragments=protein, use_standard_potentials=True,
                      standard_potential_model='ff03')
    project.create_embedding_potential(system)
    project.write_potential(system)
    assert os.path.isfile('{0}/{1}/{1}.pot'.format(tests_dir, test))
    assert filecmp.cmp('{0}/{1}/{1}.pot'.format(tests_dir, test), '{0}/{1}/{1}.pot.ref'.format(tests_dir, test))
    os.remove('{0}/{1}/{1}.pot'.format(tests_dir, test))


def test_4val():
    test = '4VAL'
    tests_dir = '{0}'.format(os.path.dirname(__file__))
    project = pyframe.Project(work_dir='{0}'.format(tests_dir))
    system = pyframe.MolecularSystem(input_file='{0}/{1}/{1}.pdb'.format(tests_dir, test), bond_threshold=1.15)
    mfcc = system.get_fragments_by_chain_id('A')
    system.add_region(name='mfcc', fragments=mfcc, use_mfcc=True,
                      use_multipoles=True, multipole_order=2,
                      use_polarizabilities=True)
    pfp = system.get_fragments_by_chain_id(chain_ids=['B'])
    system.add_region(name='pfp', fragments=pfp, use_standard_potentials=True,
                      standard_potential_model='PFP')
    project.create_embedding_potential(system)
    project.write_potential(system)
    assert os.path.isfile('{0}/{1}/{1}.pot'.format(tests_dir, test))
    assert filecmp.cmp('{0}/{1}/{1}.pot'.format(tests_dir, test), '{0}/{1}/{1}.pot.ref'.format(tests_dir, test))
    os.remove('{0}/{1}/{1}.pot'.format(tests_dir, test))


def test_popc():
    test = 'popc'
    tests_dir = '{0}'.format(os.path.dirname(__file__))
    project = pyframe.Project(work_dir='{0}'.format(tests_dir))
    system = pyframe.MolecularSystem(input_file='{0}/{1}/{1}.pdb'.format(tests_dir, test), bond_threshold=1.15)
    system.split_fragment_by_name(name='POPC', new_names=['POCH', 'POCO', 'POCP'],
                                  fragment_definitions=[['N',
                                                         'C13', 'H13A', 'H13B', 'H13C',
                                                         'C14', 'H14A', 'H14B', 'H14C',
                                                         'C15', 'H15A', 'H15B', 'H15C',
                                                         'C12', 'H12A', 'H12B',
                                                         'C11', 'H11A', 'H11B',
                                                         'P', 'O11', 'O12', 'O13', 'O14',
                                                         'C1', 'HA', 'HB',
                                                         'C2', 'HS',
                                                         'O21', 'C21', 'O22',
                                                         'C3', 'HX', 'HY',
                                                         'O31', 'C31', 'O32'],
                                                        ['C22', 'H2R', 'H2S',
                                                         'C23', 'H3R', 'H3S',
                                                         'C24', 'H4R', 'H4S',
                                                         'C25', 'H5R', 'H5S',
                                                         'C26', 'H6R', 'H6S',
                                                         'C27', 'H7R', 'H7S',
                                                         'C28', 'H8R', 'H8S',
                                                         'C29', 'H91',
                                                         'C210', 'H101',
                                                         'C211', 'H11R', 'H11S',
                                                         'C212', 'H12R', 'H12S',
                                                         'C213', 'H13R', 'H13S',
                                                         'C214', 'H14R', 'H14S',
                                                         'C215', 'H15R', 'H15S',
                                                         'C216', 'H16R', 'H16S',
                                                         'C217', 'H17R', 'H17S',
                                                         'C218', 'H18R', 'H18S', 'H18T'],
                                                        ['C32', 'H2X', 'H2Y',
                                                         'C33', 'H3X', 'H3Y',
                                                         'C34', 'H4X', 'H4Y',
                                                         'C35', 'H5X', 'H5Y',
                                                         'C36', 'H6X', 'H6Y',
                                                         'C37', 'H7X', 'H7Y',
                                                         'C38', 'H8X', 'H8Y',
                                                         'C39', 'H9X', 'H9Y',
                                                         'C310', 'H10X', 'H10Y',
                                                         'C311', 'H11X', 'H11Y',
                                                         'C312', 'H12X', 'H12Y',
                                                         'C313', 'H13X', 'H13Y',
                                                         'C314', 'H14X', 'H14Y',
                                                         'C315', 'H15X', 'H15Y',
                                                         'C316', 'H16X', 'H16Y', 'H16Z']])

    lipid = system.get_fragments_by_name(names=['POCH', 'POCO', 'POCP'])
    system.add_region(name='lipid', fragments=lipid, use_standard_potentials=True,
                      standard_potential_model='ALEP', standard_potential_exclusion_type='mfcc',
                      mfcc_order=3)
    project.create_embedding_potential(system)
    project.write_potential(system)
    assert os.path.isfile('{0}/{1}/{1}.pot'.format(tests_dir, test))
    assert filecmp.cmp('{0}/{1}/{1}.pot'.format(tests_dir, test), '{0}/{1}/{1}.pot.ref'.format(tests_dir, test))
    os.remove('{0}/{1}/{1}.pot'.format(tests_dir, test))


def test_vvv():
    test = 'VVV'
    tests_dir = f'{os.path.dirname(__file__)}'
    project = pyframe.Project(work_dir=f'{tests_dir}')
    system = pyframe.MolecularSystem(input_file=f'{tests_dir}/{test}/{test}.pdb')
    core = system.get_fragments_by_identifier(['1_VAL', '3_VAL'])
    system.set_core_region(core)
    protein = system.get_fragments_by_identifier('2_VAL')
    system.add_region(name='protein', fragments=protein, use_mfcc=True, use_multipoles=True,
                      use_polarizabilities=True, multipole_order=2)
    project.create_embedding_potential(system)
    project.write_potential(system)
    assert os.path.isfile('{0}/{1}/{1}.pot'.format(tests_dir, test))
    assert filecmp.cmp('{0}/{1}/{1}.pot'.format(tests_dir, test), '{0}/{1}/{1}.pot.ref'.format(tests_dir, test))
    os.remove('{0}/{1}/{1}.pot'.format(tests_dir, test))
    project.write_core(system)
    assert os.path.isfile('{0}/{1}/{1}.mol'.format(tests_dir, test))
    assert filecmp.cmp('{0}/{1}/{1}.mol'.format(tests_dir, test), '{0}/{1}/{1}.mol.ref'.format(tests_dir, test))
    os.remove('{0}/{1}/{1}.mol'.format(tests_dir, test))
