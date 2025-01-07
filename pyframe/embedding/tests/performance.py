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
# # print(env.multipole_fields)
# print("Calculate induced dipoles without external fields.")
# start_time = time.time()
# # print(env.self_energy)
# # # print(env.self_energy)
# # env.solve_induced_dipoles()
# # nuc_fields = core.compute_nuclear_fields(env.coordinates)
# end_time = time.time()
# print("Execution time:", end_time - start_time)

# print("Calculate induced dipoles without external fields.")
# start_time = time.time()
# env.solve_induced_dipoles()
# end_time = time.time()
# print("Execution time:", end_time - start_time)

# Print number of iterations
# print(env.induced_dipoles.number_of_iterations)
# # Print induced dipoles
# print(env.induced_dipoles.induced_dipoles)

ref_dipoles_act_wat_mid = np.array(
    [[-5.33083758e-03, 1.80899242e-02, -4.67124873e-02],
     [-6.25853096e-03, 7.25840460e-03, -1.03916985e-02],
     [-2.48028626e-02, 2.09508571e-02, -5.74245555e-02],
     [-4.03295102e-02, -2.81401191e-02, -3.81210336e-02],
     [-4.06230086e-02, -1.38089059e-02, -4.50891871e-02],
     [-4.09294307e-05, -7.71654933e-03, -1.10062634e-02],
     [-1.85295339e-02, 2.86513856e-02, -6.41668069e-03],
     [-1.89700263e-02, 3.32285459e-02, -1.05088196e-02],
     [-4.44393262e-03, 6.60779904e-03, -6.06423064e-03],
     [3.35370357e-02, -1.38589706e-02, -1.33829243e-01],
     [8.46508800e-03, -1.95121184e-02, -4.13855335e-02],
     [9.41524338e-02, -7.63125728e-03, -1.05125435e-01],
     [-3.51664680e-02, -9.46800752e-02, 3.21263242e-02],
     [-3.05118301e-02, -5.72413535e-02, -7.15883504e-03],
     [-6.53756795e-03, -1.78294688e-02, 1.00302692e-02],
     [-5.47042853e-02, -2.59646233e-02, 7.46069721e-02],
     [-1.21777884e-02, -7.14485835e-03, 1.14801021e-02],
     [-4.89534262e-03, 8.89451393e-04, 3.05621760e-02],
     [-2.45470732e-02, 3.50936051e-02, 1.62710897e-02],
     [-7.08697146e-03, 3.29550988e-02, 2.01553381e-02],
     [-3.98872500e-02, 2.68723152e-03, 2.96662818e-02],
     [1.58478302e-01, 1.75607576e-02, -9.62387317e-02],
     [3.13149664e-02, 1.50598941e-02, 3.26735356e-03],
     [1.74549438e-02, -5.22481121e-03, -1.95949401e-02],
     [5.80919877e-03, 2.57580222e-02, 6.93423314e-03],
     [3.11775698e-03, 6.69825941e-03, 2.00727748e-03],
     [2.62073304e-03, 9.94453919e-03, 7.86800192e-03],
     [-1.36831664e-01, 1.06785026e-04, -5.64310721e-02],
     [-4.13380289e-02, -1.98769850e-03, -7.29442552e-03],
     [-3.42883704e-02, -3.90445715e-03, -1.31903698e-02],
     [-3.98766230e-02, 1.50048575e-02, 4.10491141e-02],
     [-7.03320227e-03, 3.14323320e-03, 9.82858536e-03],
     [-3.91029547e-02, -2.50420088e-02, 4.75975138e-02],
     [-3.19239004e-02, -3.78333796e-02, 1.03070603e-02],
     [-7.28289756e-03, -1.31227630e-02, 6.04312467e-04],
     [-2.39672048e-02, -2.77544312e-02, 1.05652401e-02],
     [-1.69413808e-02, 1.32959670e-02, 4.20292421e-02],
     [5.80488730e-03, 1.17613126e-02, 1.39260906e-02],
     [-1.40970191e-02, 8.70572639e-03, 1.27696986e-02],
     [-2.66591757e-02, -8.20268170e-02, -1.42113498e-02],
     [1.23204937e-03, -2.97407616e-02, 2.68551812e-03],
     [-3.37210637e-04, -1.45256899e-02, -3.48008725e-03]])

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
# print(env.coordinates[0:1])
print(core.compute_nuclear_fields(env.coordinates[0:1]))
# for i in range(10000):
#     env.compute_repulsion_energy('Lorentz-Berthelot')

# print(env.compute_dispersion_energy('Lorentz-Berthelot') - -0.00131364768168478)
end_time = time.time()
print("Execution time:", end_time - start_time)
