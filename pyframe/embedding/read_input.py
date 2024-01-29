from __future__ import annotations

import json
import numpy as np

from typing import Optional
from pyframe.embedding import fragment, particle, density_matrix, subsystem
from pathlib import Path

# input dictionary either from json or directly (two functions) returns quantum subsystem or list of classical subsystems.

def json_to_dict(filepath: Path | str) -> dict:
    try:
        with open(filepath, 'r') as json_file:
            data_dictionary = json.load(json_file)
        return data_dictionary
    except FileNotFoundError:
        print(f"Error: File not found at path {filepath}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file at path {filepath}")
        return {}
def reader(input_data: dict | Path | str):
    # TODO option to only return QM or MM subsystem
    # what does a quantum subsystem want? list of nuclei  optional: name, density_matrix (has to be updated later then), quantum_fragments
    if isinstance(input_data, dict):
        print("Creating Quantum Subsystem and Classical Subsystem from dictionary.")
    if isinstance(input_data, Path):
        print("Creating Quantum Subsystem and Classical Subsystem from Path object.")
        input_data = json_to_dict(input_data)
    elif isinstance(input_data, str):
        print("Creating Quantum Subsystem and Classical Subsystem from string path.")
        input_data = json_to_dict(input_data)
    else:
        raise TypeError("Input data has an unrecognized type.")
    quantum_fragments = None
    q_name = None
    c_name = None
    dens_mat = density_matrix.DensityMatrix(np.zeros(1))
    # Quantum Subsystem
    if input_data.get('quantum_subsystem', None) is None:
        raise KeyError("There is no Quantum Subsystem.")
    quantum_subsystem_data = input_data.get('quantum_subsystem', None)
    if quantum_subsystem_data.get('name', None) is not None:
        q_name = quantum_subsystem_data['name']
    if quantum_subsystem_data.get('nuclei', None) is not None:
        nuclei = []
        for nuc in quantum_subsystem_data['nuclei']:
            nuc['coordinate'] = np.array(nuc['coordinate'])
            nuclei.append(particle.Nucleus(**nuc))
    else:
        raise KeyError("Nuclei are not present in the Quantum Subsystem.")
    if quantum_subsystem_data.get('quantum_fragments', None) is not None:
        quantum_fragments = []
        for frag in quantum_subsystem_data['quantum_fragments']:
            quantum_fragments.append(fragment.QuantumFragment(**frag))
    if quantum_subsystem_data.get('density_matrix', None) is not None:
        dens_mat = density_matrix.DensityMatrix(quantum_subsystem_data['density_matrix'])
    # Classical Subsystem
    if input_data.get('classical_subsystem', None) is None:
        raise KeyError("There is no Classical Subsystem.")
    classical_subsystem_data = input_data.get('classical_subsystem', None)
    if isinstance(classical_subsystem_data, list):
        classical_subsystems = []
        for c_subsystem in classical_subsystem_data:
            if c_subsystem.get('classical_fragments', None) is not None:
                classical_fragments = []
                for f in c_subsystem['classical_fragments']:
                    classical_fragments.append(fragment.ClassicalFragment(**f))
            else:
                raise KeyError("Fragments are not present in the Classical Subsystem.")
            if c_subsystem.get('name', None) is not None:
                c_name = quantum_subsystem_data['name']
            classical_subsystems.append(subsystem.ClassicalSubsystem(classical_fragments=classical_fragments,
                                                                     name=c_name))
        return (subsystem.QuantumSubsystem(nuclei=nuclei, dens_mat=dens_mat, quantum_fragments=quantum_fragments,
                                           name=q_name),
                classical_subsystems)
    if isinstance(classical_subsystem_data, dict):
        if classical_subsystem_data.get('classical_fragments', None) is not None:
            classical_fragments = []
            for f in classical_subsystem_data['classical_fragments']:
                classical_fragments.append(fragment.ClassicalFragment(**f))
        else:
            raise KeyError("Fragments are not present in the Classical Subsystem.")
        if classical_subsystem_data.get('name', None) is not None:
            c_name = quantum_subsystem_data['name']
        return  (subsystem.QuantumSubsystem(nuclei=nuclei, dens_mat=dens_mat, quantum_fragments=quantum_fragments,
                                            name=q_name),
                 subsystem.ClassicalSubsystem(classical_fragments=classical_fragments, name=c_name))

    # input_data dict keys: "quantum_subsystem" and "classical_subsystem"
    # one layer below: for quantum_subsystem: "name", "comment", "density_matrix", "nuclei", ("quantum_fragments")
    # one layer below: for classical_subsystem: "name", "comment", "classical_fragments" -> particles, multipoles, polarizabilities etc.