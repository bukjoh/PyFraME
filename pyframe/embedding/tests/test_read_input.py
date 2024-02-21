"""Tests PyFraME.embedding.read_input.py"""
import pytest
import json
import os

from pathlib import Path
from pyframe.embedding import read_input, subsystem


@pytest.fixture
def sample_json(tmp_path):
    # Create a sample JSON file for testing
    data = {
        "quantum_subsystem": {
            "name": "quantum",
            "nuclei": [{"coordinate": [3, 0, 0], "name": "nucleus1", "index": 1, "charge": 1.0}],
            "quantum_fragments": [{"name": "fragment1",
                                   "index": 1,
                                   "nuclei": [{"coordinate": [0, 0, 2], "name": "nucleus1", "index": 1, "charge": 1.0}],
                                   "e_density_matrix": [0, 0, 0, 0]}],
            "density_matrix": [1, 0, 0, 1]
        },
        "classical_subsystem": {
            "name": "classical",
            "classical_fragments": [{"name": "classical_fragment1",
                                     'index': 6,
                                     'atoms': [{'element': 'H', 'coordinate': [0, 5, 0], 'index': 0,
                                                'multipoles': {"elements": [0]},
                                                'polarizabilities': {"elements": [0], "order": [0]},
                                                'exclusions': [0]},
                                               {'element': 'O', 'coordinate': [1, 2, 1], 'index': 1,
                                                'multipoles': {"elements": [0]},
                                                'polarizabilities': {"elements": [0], "order": [0]},
                                                'exclusions': [1]
                                                }]
                                     }]
        }
    }
    file_path = tmp_path / "test.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    return file_path


@pytest.fixture
def sample_json_path():
    # Construct the path to the sample JSON file
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, "data", "two_atom_test.json")


def test_dict_input(sample_json_path):
    # Test if reader function works with dictionary input
    with open(sample_json_path, 'r') as file:
        input_data = json.load(file)
    quantum_subsystem, classical_subsystem = read_input.reader(input_data)
    assert isinstance(quantum_subsystem, subsystem.QuantumSubsystem)
    assert isinstance(classical_subsystem, subsystem.ClassicalSubsystem)


def test_path_input(sample_json_path):
    # Test if reader function works with Path object input
    input_data = Path(sample_json_path)
    quantum_subsystem, classical_subsystem = read_input.reader(input_data)
    assert isinstance(quantum_subsystem, subsystem.QuantumSubsystem)
    assert isinstance(classical_subsystem, subsystem.ClassicalSubsystem)


def test_string_input(sample_json_path):
    # Test if reader function works with string path input
    input_data = str(sample_json_path)
    quantum_subsystem, classical_subsystem = read_input.reader(input_data)
    assert isinstance(quantum_subsystem, subsystem.QuantumSubsystem)
    assert isinstance(classical_subsystem, subsystem.ClassicalSubsystem)


def test_unrecognized_input():
    # Test if reader function raises TypeError for unrecognized input type
    with pytest.raises(TypeError, match="Input data has an unrecognized type."):
        read_input.reader(123)  # Passing an integer as input data


def test_json_to_dict(sample_json):
    # Test if json_to_dict function works as expected
    data = read_input.json_to_dict(sample_json)
    assert isinstance(data, dict)
    assert "quantum_subsystem" in data
    assert "classical_subsystem" in data


def test_reader(sample_json):
    # Test if reader function works as expected
    quantum_subsystem, classical_subsystem = read_input.reader(sample_json)
    assert quantum_subsystem.name == "quantum"
    assert quantum_subsystem.nuclei[0].name == "nucleus1"
    assert quantum_subsystem.nuclei[0].index == 1
    assert quantum_subsystem.nuclei[0].charge == 1.0
    assert quantum_subsystem.quantum_fragments[0]._name == "fragment1"
    assert classical_subsystem.name == "classical"
    assert classical_subsystem.classical_fragments[0]._name == "classical_fragment1"
    assert classical_subsystem.classical_fragments[0]._index == 6


def test_missing_quantum_subsystem():
    # Test if KeyError is raised when quantum_subsystem key is missing
    input_data = f'{os.path.dirname(__file__)}/data/only_classical_subsystem.json'
    with pytest.raises(KeyError, match="There is no Quantum Subsystem."):
        read_input.reader(input_data)
    classical_subsystem = read_input.reader(input_data,
                                            read_quantum=False,
                                            read_classical=True)
    assert isinstance(classical_subsystem, subsystem.ClassicalSubsystem)


def test_missing_classical_subsystem():
    # Test if KeyError is raised when classical_subsystem key is missing
    input_data = f'{os.path.dirname(__file__)}/data/only_quantum_subsystem.json'
    with pytest.raises(KeyError, match="There is no Classical Subsystem."):
        read_input.reader(input_data)
    quantum_subsystem = read_input.reader(input_data,
                                          read_classical=False,
                                          read_quantum=True)
    assert isinstance(quantum_subsystem, subsystem.QuantumSubsystem)


def test_missing_nuclei_key():
    # Test if KeyError is raised when nuclei key is missing in quantum_subsystem
    input_data = f'{os.path.dirname(__file__)}/data/missing_nuclei_two_atoms_test.json'
    with pytest.raises(KeyError, match="Nuclei are not present in the Quantum Subsystem."):
        read_input.reader(input_data)


def test_missing_fragments_key():
    # Test if KeyError is raised when quantum_fragments or classical_fragments key is missing
    input_data = f'{os.path.dirname(__file__)}/data/missing_classical_fragments_two_atoms_test.json'
    with pytest.raises(KeyError, match="Fragments are not present in the Classical Subsystem."):
        read_input.reader(input_data)


def test_both_read_flags_false():
    # Test if KeyError is raised when both read flags are set to False
    with pytest.raises(KeyError, match="Both read flags are set to False, nothing is read in."):
        read_input.reader(f'{os.path.dirname(__file__)}/data/two_atom_test.json',
                          read_quantum=False,
                          read_classical=False)
