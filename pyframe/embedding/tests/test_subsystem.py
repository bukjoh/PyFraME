"""Tests PyFraME.embedding.subsystem.py"""
import os
from pyframe.embedding import subsystem
from qcelemental import PhysicalConstantsContext

constants = PhysicalConstantsContext('CODATA2018')


def test_init_subsystem():
    subsystem.Subsystem(name="2x H2O fragments, O and H particles")


def test_init_classical_subsystem():
    subsystem.ClassicalSubsystem(name="2x H2O fragments, O and H particles",
                                 input_data=f'{os.path.dirname(__file__)}/data/act_wat_test.json')


def test_init_quantum_subsystem():
    subsystem.QuantumSubsystem(name="2x H2O fragments, O and H particles, H2O density",
                               input_data=f'{os.path.dirname(__file__)}/data/act_wat_test.json')
