#!/bin/bash

for NT in {1..16}; do julia --project -O3 -t $NT LAPACK-threads_benchmark.jl | tee -a parallel-LAPACK.log; done
