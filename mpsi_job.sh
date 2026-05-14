#!/bin/bash
#$ -pe mpi 4
# qsub mpi_job.sh

mpirun -np 4 python montecarlo_mpi.py
