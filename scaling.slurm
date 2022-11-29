#!/bin/bash -l

#SBATCH --partition=medium
#SBATCH --mem=128gb
#SBATCH --time=12:00:00
#SBATCH --nodes=4
#SBATCH --tasks-per-node=44
#SBATCH --job-name=pdgemv-scaling
#SBATCH --output=pdgemv-scaling.%j.out
#SBATCH --error=pdgemv-scaling.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=troupb18@wfu.edu

mpirun -n 1 ./a.out 17600 17600 1 1 yes
mpirun -n 2 ./a.out 17600 17600 2 1 yes
mpirun -n 4 ./a.out 17600 17600 2 2 yes
mpirun -n 5 ./a.out 17600 17600 5 1 yes
mpirun -n 8 ./a.out 17600 17600 4 2 yes
mpirun -n 10 ./a.out 17600 17600 5 2 yes
mpirun -n 11 ./a.out 17600 17600 11 1 yes
mpirun -n 16 ./a.out 17600 17600 4 4 yes
mpirun -n 20 ./a.out 17600 17600 5 4 yes
mpirun -n 22 ./a.out 17600 17600 11 2 yes
mpirun -n 25 ./a.out 17600 17600 5 5 yes
mpirun -n 32 ./a.out 17600 17600 16 2 yes
mpirun -n 40 ./a.out 17600 17600 10 4 yes
mpirun -n 44 ./a.out 17600 17600 11 4 yes
mpirun -n 50 ./a.out 17600 17600 25 2 yes
mpirun -n 55 ./a.out 17600 17600 11 5 yes
mpirun -n 64 ./a.out 17600 17600 16 4 yes
mpirun -n 80 ./a.out 17600 17600 10 8 yes
mpirun -n 88 ./a.out 17600 17600 11 8 yes
mpirun -n 100 ./a.out 17600 17600 10 10 yes
mpirun -n 110 ./a.out 17600 17600 10 11 yes
mpirun -n 125 ./a.out 17600 17600 25 5 yes
mpirun -n 128 ./a.out 17600 17600 16 8 yes
mpirun -n 160 ./a.out 17600 17600 16 10 yes
mpirun -n 176 ./a.out 17600 17600 16 11 yes

