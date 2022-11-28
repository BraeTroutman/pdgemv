#!/bin/bash

run_and_print () {
	p=$1
	rows=$2
	cols=$3
	pr=$4
	pc=$5
	echo "**************************************************************"
	echo "Multiplying $rows x $cols matrix with $pr x $pc processor grid"
	mpirun -n $p ./a.out $rows $cols $pr $pc yes
}

run_and_print 1 6 6 1 1
run_and_print 2 6 6 2 1
run_and_print 2 6 6 1 2
run_and_print 6 6 6 3 2
run_and_print 6 6 6 2 3
run_and_print 1 100 100 1 1
run_and_print 2 100 100 2 1
run_and_print 2 100 100 1 2
run_and_print 10 100 100 5 2
run_and_print 10 100 100 2 5
run_and_print 10 100 100 1 10
run_and_print 10 100 100 10 1
run_and_print 1 1000 1000 1 1
run_and_print 20 1000 1000 20 1
run_and_print 20 1000 1000 1 20
run_and_print 20 1000 1000 5 4
run_and_print 20 1000 1000 4 5

