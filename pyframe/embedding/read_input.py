from __future__ import annotations

import json
import numpy as np

from pyframe.embedding import fragment, particle, density_matrix, subsystem
from pathlib import Path
from typing import List, Tuple, Optional
from mpi4py import MPI


def json_to_dict(filepath: Path | str
                 ) -> dict:
    """Converts a JSON file to a Python dictionary.
    file.

    Args:
        filepath: Path object or the string of the path to the JSON file.

    Returns:
        Dictionary of data in JSON file.

    """
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


def reader(input_data: dict | Path | str,
           read_quantum: Optional[bool] = True,
           read_classical: Optional[bool] = True,
           comm: Optional[MPI.Comm] = None
           ) -> (Tuple[subsystem.QuantumSubsystem, subsystem.ClassicalSubsystem] |
                 Tuple[subsystem.QuantumSubsystem, List[subsystem.ClassicalSubsystem]] |
                 subsystem.QuantumSubsystem |
                 subsystem.ClassicalSubsystem |
                 List[subsystem.ClassicalSubsystem]):
    """Reads in a JSON file or Python dictionary and creates instances of a QuantumSubsystem, ClassicalSubsystem(s) or
     both.

    Args:
        input_data: JSON file or dictionary containing the data.
        read_quantum: Flag to indicate if instance of QuantumSubsystem is to be read in and created.
        read_classical: Flag to indicate if instance or list of instances of ClassicalSubsystem(s) is to be read in and
        created.
        comm: The MPI communicator.

    Returns:
        QuantumSubsystem, ClassicalSubsystem(s) or both.
    """
    if isinstance(input_data, dict):
        print("Creating from dictionary.")
    elif isinstance(input_data, Path):
        print("Creating from Path object.")
        input_data = json_to_dict(input_data)
        if not bool(input_data):
            raise RuntimeError("Input data not created successfully, please check filepath.")
    elif isinstance(input_data, str):
        print("Creating from string path.")
        input_data = json_to_dict(input_data)
        if not bool(input_data):
            raise RuntimeError("Input data not created successfully, please check filepath.")
    else:
        raise TypeError("Input data has an unrecognized type.")
    if not read_quantum and not read_classical:
        raise KeyError("Both read flags are set to False, nothing is read in.")
    quantum_fragments = None
    q_name = None
    c_name = None
    quantum_subsystem = None
    classical_subsystem = None
    classical_subsystems = None
    dens_mat = density_matrix.DensityMatrix(np.zeros(1))
    # Quantum Subsystem
    if read_quantum:
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
        quantum_subsystem = subsystem.QuantumSubsystem(nuclei=nuclei,
                                                       dens_mat=dens_mat,
                                                       quantum_fragments=quantum_fragments,
                                                       name=q_name,
                                                       comm=comm)
    # TODO make the JSON input to 'classical_subsystems' and always give it as a list.
    # TODO and instead make keyword 'classical_subsystem' always as a singular dictionary.

    # Classical Subsystem
    if read_classical:
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
                    c_name = c_subsystem['name']
                classical_subsystems.append(subsystem.ClassicalSubsystem(classical_fragments=classical_fragments,
                                                                         name=c_name,
                                                                         comm=comm))
        if isinstance(classical_subsystem_data, dict):
            if classical_subsystem_data.get('classical_fragments', None) is not None:
                classical_fragments = []
                for f in classical_subsystem_data['classical_fragments']:
                    classical_fragments.append(fragment.ClassicalFragment(**f))
            else:
                raise KeyError("Fragments are not present in the Classical Subsystem.")
            if classical_subsystem_data.get('name', None) is not None:
                c_name = classical_subsystem_data['name']
            classical_subsystem = subsystem.ClassicalSubsystem(classical_fragments=classical_fragments,
                                                               name=c_name,
                                                               comm=comm)

    if quantum_subsystem is not None:
        if classical_subsystem is not None:
            return quantum_subsystem, classical_subsystem
        elif classical_subsystems is not None:
            return quantum_subsystem, classical_subsystems
        else:
            return quantum_subsystem
    else:
        if classical_subsystem is not None:
            return classical_subsystem
        elif classical_subsystems is not None:
            return classical_subsystems
