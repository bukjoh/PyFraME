"""Tests PyFraME.embedding.subsystem.py"""
import os
import numpy as np
from pyframe.embedding import read_input
from qcelemental import PhysicalConstantsContext

constants = PhysicalConstantsContext('CODATA2018')

# should not test init put potential + total energy?!
# TODO

core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_test.json')

def test_compute_nuclear_fields():
    ref_array = np.array([[0.00392044, -0.05293017, -0.10427627],
                          [0.00469095, -0.05492494, -0.11713605],
                          [0.00066334, -0.04590806, -0.09858109],
                          [-0.26160936, 0.11023125, 0.08629496],
                          [-0.26141526, 0.0838028 , 0.09513431],
                          [-0.27319751, 0.13408809, 0.1133901 ]])
    assert np.allclose(core.compute_nuclear_fields(env.coordinates), ref_array)
    assert np.all(np.isnan(core.compute_nuclear_fields(core.coordinates)) == True)

#def test_compute_electric_fields():

