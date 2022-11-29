#!/bin/bash -l

#SBATCH --partition=small
#SBATCH --mem=128gb
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=44
#SBATCH --job-name=bottleneck
#SBATCH --output=bottleneck.%j.out
#SBATCH --error=bottleneck.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=troupb18@wfu.edu

for ((i=108; i<=100000; i+=360))
do
	mpirun -n 36 ./a.out $i $i 6 6 yes
done

