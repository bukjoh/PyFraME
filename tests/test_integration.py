# coding=utf-8
"""Integration tests"""

import unittest
import os
import filecmp

import pyframe
import pyframe.readers

class IntegrationTests(unittest.TestCase):

    def test_permanganate(self):
        test = 'permanganate'
        tests_dir = '{0}'.format(os.path.dirname(__file__))
        project = pyframe.Project(work_dir='{0}'.format(tests_dir))
        system = pyframe.MolecularSystem(input_file='{0}/{1}/{1}.pdb'.format(tests_dir, test), bond_threshold=1.15)
        core = system.get_fragments_by_name(names=['LIG'])
        system.set_core_region(core)
        solvent = system.get_fragments_by_name(names=['HOH'])
        system.add_region(name='solvent', fragments=solvent, use_standard_potentials=True, standard_potential_model='TIP3P')
        project.create_embedding_potential(system)
        project.write_core(system)
        self.assertTrue(filecmp.cmp('{0}/{1}/{1}.mol'.format(tests_dir, test), '{0}/{1}/{1}.mol.ref'.format(tests_dir, test)))
        project.write_potential(system)
        self.assertTrue(filecmp.cmp('{0}/{1}/{1}.pot'.format(tests_dir, test), '{0}/{1}/{1}.pot.ref'.format(tests_dir, test)))

    def test_pna_in_ccl4(self):
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
        self.assertTrue(filecmp.cmp('{0}/{1}/{1}.mol'.format(tests_dir, test), '{0}/{1}/{1}.mol.ref'.format(tests_dir, test)))
        project.write_potential(system)
        self.assertTrue(filecmp.cmp('{0}/{1}/{1}.pot'.format(tests_dir, test), '{0}/{1}/{1}.pot.ref'.format(tests_dir, test)))

    def test_insulin(self):
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
        potential = pyframe.readers.read_pelib_potential('{0}/{1}/{1}.pot'.format(tests_dir, test))
        reference_potential = pyframe.readers.read_pelib_potential('{0}/{1}/{1}.pot.ref'.format(tests_dir, test))
        for site, ref_site in zip(potential.values(), reference_potential.values()):
            self.assertEqual(site.element, ref_site.element)
            for comp, ref_comp in zip(site.coordinate, ref_site.coordinate):
                self.assertAlmostEqual(comp, ref_comp)
            for comp, ref_comp in zip(site.M0, ref_site.M0):
                self.assertAlmostEqual(comp, ref_comp)
            for comp, ref_comp in zip(site.M1, ref_site.M1):
                self.assertAlmostEqual(comp, ref_comp)
            for comp, ref_comp in zip(site.M2, ref_site.M2):
                self.assertAlmostEqual(comp, ref_comp)
            for comp, ref_comp in zip(site.P11, ref_site.P11):
                self.assertAlmostEqual(comp, ref_comp)
