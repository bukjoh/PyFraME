from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from mpi4py import MPI

from .fragment import QuantumFragment, ClassicalFragment
from .particle import Nucleus
from .subsystem import QuantumSubsystem, ClassicalSubsystem
from pyframe.simulation_box import SimulationBox
from .logging_util import log_manager

__all__ = 'reader'


def json_to_dict(filepath: Path | str
                 ) -> dict:
    """Convert a JSON file to a Python dictionary.
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
           comm: MPI.Comm | None = None
           ) -> (tuple[QuantumSubsystem, ...] | tuple[ClassicalSubsystem, ...] |
                 tuple[QuantumSubsystem, ..., ClassicalSubsystem, ..., SimulationBox]):
    """Read in a JSON file or Python dictionary and create instances of a QuantumSubsystem, ClassicalSubsystem(s) or
     both.

    Args:
        input_data: Path to a JSON input file, the name of a JSON input file, or a dictionary containing the
                subsystem(s) data.
        comm: The MPI communicator.

    Returns:
        QuantumSubsystem(s) and/ or ClassicalSubsystem(s).
    """
    if isinstance(input_data, dict):
        log_manager.logger.debug(
            print("Creating subsystem(s) from input dictionary.")
        )
        subsystems_data = input_data
    elif isinstance(input_data, Path) or isinstance(input_data, str):
        log_manager.logger.debug(
            print("Creating from Path or str.")
        )
        subsystems_data = json_to_dict(input_data)
        if not input_data:
            raise RuntimeError("Input data not created successfully, please check filepath.")
    else:
        raise TypeError(f'Argument "input_data" must be of type "dict", "Path", or "str" '
                        f'(not "{type(input_data).__name__}").')
    system = []
    quantum_fragments = None
    quantum_subsystem_name = None
    classical_subsystem_name = None
    # system data
    if subsystems_data.get('system', None) is not None:
        system_data = subsystems_data.get('system', None)
        # Assuming system_data is a list, get the first dictionary element
        if isinstance(system_data, list) and len(system_data) > 0:
            box = system_data[0].get('simulation_box', [{}])[0]
            # Extract dimensions and angles
            lengths = box.get('lengths', [])
            angles = box.get('angles', [])
            system.append(SimulationBox(lengths=lengths,
                                        angles=angles))
    # Quantum Subsystems
    if subsystems_data.get('quantum_subsystems', None) is not None:
        quantum_subsystems = subsystems_data.get('quantum_subsystems', None)
        for quantum_subsystem_data in quantum_subsystems:
            if quantum_subsystem_data.get('name', None) is not None:
                quantum_subsystem_name = quantum_subsystem_data['name']
            if quantum_subsystem_data.get('nuclei', None) is not None:
                nuclei = []
                for nucleus in quantum_subsystem_data['nuclei']:
                    nucleus['coordinate'] = np.array(nucleus['coordinate'])
                    nuclei.append(Nucleus(**nucleus))
            else:
                raise KeyError("Nuclei are not present in the Quantum Subsystem.")
            if quantum_subsystem_data.get('quantum_fragments', None) is not None:
                quantum_fragments = []
                for fragment in quantum_subsystem_data['quantum_fragments']:
                    quantum_fragments.append(QuantumFragment(**fragment))
            system.append(QuantumSubsystem(nuclei=nuclei,
                                           quantum_fragments=quantum_fragments,
                                           name=quantum_subsystem_name,
                                           comm=comm))
    # Classical Subsystems
    if subsystems_data.get('classical_subsystems', None) is not None:
        classical_subsystems = subsystems_data.get('classical_subsystems', None)
        for classical_subsystem_data in classical_subsystems:
            if classical_subsystem_data.get('classical_fragments', None) is not None:
                classical_fragments = []
                for f in classical_subsystem_data['classical_fragments']:
                    classical_fragments.append(ClassicalFragment(**f))
            else:
                raise KeyError("Fragments are not present in the Classical Subsystem.")
            if classical_subsystem_data.get('name', None) is not None:
                classical_subsystem_name = classical_subsystem_data['name']
            system.append(ClassicalSubsystem(classical_fragments=classical_fragments,
                                             name=classical_subsystem_name,
                                             comm=comm))
    if not system:
        raise KeyError("No Subsystems could be read in.")
    return tuple(system)
