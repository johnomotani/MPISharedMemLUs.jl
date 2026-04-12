#!/bin/bash

for NP in {1..16}; do mpirun -np $NP julia --project -O3 -t 1 MPI-ColumnPivotLUs_benchmark.jl | tee -a parallel-ColumnPivotLUs.log; done
