"""Tests PyFraME.embedding.subsystem.py"""
import os
from pyframe.embedding import subsystem, read_input
from qcelemental import PhysicalConstantsContext

constants = PhysicalConstantsContext('CODATA2018')

# should not test init put potential + total energy?!
# TODO
def test_init_subsystem():
    subsystem.Subsystem(name="2x H2O fragments, O and H particles")


def test_init_classical_subsystem():
    b, a = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_test.json')
def test_init_quantum_subsystem():
    b, a = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_test.json')

