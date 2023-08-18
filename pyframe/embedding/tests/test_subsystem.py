"""Tests PyFraME.embedding.subsystem.py"""
import os
from pyframe.embedding import subsystem, vlx_interface, electrostatic_interactions, induction_interactions
from qcelemental import PhysicalConstantsContext

constants = PhysicalConstantsContext('CODATA2018')


def test_init_subsystem():
    subsystem.Subsystem(name="2x H2O fragments, O and H particles")


def test_init_classical_subsystem():
    a = subsystem.ClassicalSubsystem(name="2x H2O fragments, and a O particle",
                                     input_data=f'{os.path.dirname(__file__)}/data/act_wat_test.json')
    b = subsystem.QuantumSubsystem(name="2x H2O fragments, and a O particle, H2O density",
                               input_data=f'{os.path.dirname(__file__)}/data/act_wat_test.json')


    # cleanup later
    h2o_xyz = """3
    water
    O        0.0000000000      0.0000000000      0.0000000000                 
    H        0.6891400000      0.8324710000      0.0000000000                 
    H        0.7224340000     -0.8726890000      0.0000000000
    """
    basis = 'sto-3g'
    driver = vlx_interface.EmbeddingIntegralDriver(h2o_xyz, basis)
    fock_contr = electrostatic_interactions.es_fock_matrix_contributions(classical_subsystem=a, integral_drv=driver)
    #print(fock_contr)


def test_init_quantum_subsystem():
    subsystem.QuantumSubsystem(name="2x H2O fragments, and a O particle, H2O density",
                               input_data=f'{os.path.dirname(__file__)}/data/act_wat_test.json')

# test for potential missing!
