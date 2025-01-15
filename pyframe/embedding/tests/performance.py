import time
import os
import sys
import numpy as np
from mpi4py import MPI
from pyframe.embedding import read_input
import cProfile

os.environ["OMP_NUM_THREADS"] = "4"
comm = MPI.COMM_WORLD
# comm = None
# core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/nilered.json',
#                               comm=comm)
# print(core.num_nuclei)
# print(env.num_atoms)
# print(np.max(env.multipole_orders))
# # cProfile.run('env.self_energy')
#
# print(env.multipole_fields)
#print("Calculate induced dipoles without external fields.")
#start_time = time.time()
# print(env.self_energy)
# # print(env.self_energy)
# env.solve_induced_dipoles()
# nuc_fields = core.compute_nuclear_fields(env.coordinates)
#end_time = time.time()
#print("Execution time:", end_time - start_time)

solver = 'dcjidiis'

# print("Calculate induced dipoles without external fields.")
start_time = time.time()
env.solve_induced_dipoles(solver=solver, mic=True, box=simulation_box.box)
end_time = time.time()
print("Execution time:", end_time - start_time)

# print("Calculate induced dipoles without external fields.")
# start_time = time.time()
# env.solve_induced_dipoles()
# end_time = time.time()
# print("Execution time:", end_time - start_time)

# Print number of iterations
print("Number of iterations", env.induced_dipoles.number_of_iterations)
# Print induced dipoles
print("Induced dipoles:\n", env.induced_dipoles.induced_dipoles)
# print("multipoles:\n", env.degenerate_multipoles_with_taylor_coefficients)
# print("coordinates:\n", env.coordinates)
# print("Permanent dipoles:\n", env.coordinates * env.degenerate_multipoles_with_taylor_coefficients)

# ref_num_iter = 16
# ref_exec_time = 0.3774082660675049
#
# comm = None
# core_s, env_s = read_input.reader(subsystems_data=f'{os.path.dirname(__file__)}/data/act_wat_big.json',
#                                   comm=comm)
# print("Calculate induced dipoles without external fields.")
# start_time = time.time()
# env_s.solve_induced_dipoles()
# # env_s.multipole_fields
# end_time = time.time()
# print("Execution time:", end_time - start_time)
# # #
# assert np.allclose(env.induced_dipoles.induced_dipoles, env_s.induced_dipoles.induced_dipoles)
# assert env.induced_dipoles.number_of_iterations == env_s.induced_dipoles.number_of_iterations
# # # assert np.allclose(nuc_fields, nuc_s_fields)
# # assert np.allclose(env_s.multipole_fields, env.multipole_fields)
# print(env_s.induced_dipoles.number_of_iterations)
# Run in terminal to compile cpp
# python .\setup.py build_ext --inplace

core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/data/act_wat_mid.json',
                              comm=comm)
from pyframe.embedding import electrostatic_interactions
print("Calculate")
start_time = time.time()

#print(electrostatic_interactions.compute_electrostatic_nuclear_energy(quantum_subsystem=core, classical_subsystem=env))

#print(electrostatic_interactions.compute_electrostatic_nuclear_gradients(quantum_subsystem=core, classical_subsystem=env))
print(env.compute_electrostatic_energy())
# print(env.coordinates[0:1])
#print(core.compute_nuclear_fields(env.coordinates[0:1]))
# for i in range(10000):
#     env.compute_repulsion_energy('Lorentz-Berthelot')

# print(env.compute_dispersion_energy('Lorentz-Berthelot') - -0.00131364768168478)
end_time = time.time()
print("Execution time:", end_time - start_time)
