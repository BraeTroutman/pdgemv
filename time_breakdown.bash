#!/bin/bash

for ((i=112; i<=100000; i+=16))
do
	mpirun -n 16 ./a.out $i $i 4 4 yes
done

