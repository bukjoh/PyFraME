import time
import os
from mpi4py import MPI
from pyframe.embedding import read_input
# comm = MPI.COMM_WORLD
comm = None
core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid/act_wat_mid.json',
                              comm=comm)
print("Calculate induced dipoles without external fields.")
start_time = time.time()
env.solve_induced_dipoles()
end_time = time.time()
print("Execution time:", end_time - start_time)

# Print number of iterations
print(env.induced_dipoles.number_of_iterations)
# Print induced dipoles
print(env.induced_dipoles.induced_dipoles)

# Run in terminal to compile cpp
# python .\setup.py build_ext --inplace