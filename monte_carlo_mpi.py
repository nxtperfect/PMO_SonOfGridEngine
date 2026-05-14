from mpi4py import MPI
import random

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()


def f(x, y):
    return x * x + y * y


samples_per_process = 250000

partial_sum = 0.0

for _ in range(samples_per_process):
    x = random.random()
    y = random.random()

    partial_sum += f(x, y)

# collect all partial sums
total_sum = comm.reduce(partial_sum, op=MPI.SUM, root=0)

if rank == 0:
    total_samples = samples_per_process * size

    estimate = total_sum / total_samples

    print("Estimated integral:", estimate)
