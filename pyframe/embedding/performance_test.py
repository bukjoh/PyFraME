import os
import time
from pyframe.embedding import read_input
from mpi4py import MPI
import numpy as np

# FIXME every process has to create an instance of core and env, and cannot just be broadcasted by rank==0 because
#  TypeError: cannot pickle 'mpi4py.MPI.Intracomm' object
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# comm = None
core, env = read_input.reader(input_data=f'{os.path.dirname(__file__)}/tests/data/act_wat_big.json', comm=comm)
print("Calculate nuclear fields without external fields.")
start_time = time.time()
core.compute_nuclear_fields(coordinates=env.coordinates)
end_time = time.time()
print("Execution time:", end_time - start_time)
gathered_summed_arr = comm.gather(core.compute_nuclear_fields(coordinates=env.coordinates), root=0)
if rank == 0:
    for i in range(1, comm.Get_size()):
        assert np.array_equal(gathered_summed_arr[0], gathered_summed_arr[i]), "Arrays are not equal across processes"
# print("Calculate multipole fields without external fields.")
# start_time = time.time()
# env.multipole_fields
# end_time = time.time()
# print("Execution time:", end_time - start_time)
# print("Calculate induced dipoles without external fields.")
# start_time = time.time()
# env.solve_induced_dipoles()
# end_time = time.time()
# print("Execution time:", end_time - start_time)
# gathered_summed_arr = comm.gather(env.induced_dipoles.induced_dipoles, root=0)
# gathered_summed_arr2 = comm.gather(env._multipole_fields, root=0)
# # Check if the gathered arrays are equal on all processes
# if rank == 0:
#     for i in range(1, comm.Get_size()):
#         assert np.array_equal(gathered_summed_arr[0], gathered_summed_arr[i]), "Arrays are not equal across processes"
#         assert np.array_equal(gathered_summed_arr2[0], gathered_summed_arr2[i]), "Arrays are not equal across processes"
MPI.Finalize()

