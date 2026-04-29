# MPISharedMemLUs

[![Build Status](https://github.com/johnomotani/MPISharedMemLUs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/johnomotani/MPISharedMemLUs.jl/actions/workflows/CI.yml?query=branch%3Amain)

The package provides a variant of the
[LAPACK](https://github.com/Reference-LAPACK/lapack/) LU factorisation algorithm that
uses column pivoting instead of row pivoting.

A direct copy of the standard (row-pivoting) algorithm is also included to enable a fair
performance comparison.

The column-pivoting variant is less efficient (maybe ~2x), so should be used only for
special applications where column pivoting is required. Both versions support
some parallelism using MPI shared-memory arrays.

Usage
-----

In serial
```julia
using MPISharedMemLUs
using LinearAlgebra

n = 4

A = rand(n,n)

ipiv = zeros(n)
Alu_rowpivot = get_row_pivot_lu(ipiv)

lu!(Alu_rowpivot, A)

A .= rand(n,n)

jpiv = zeros(n)
Alu_colpivot = get_column_pivot_lu(jpiv)

lu!(Alu_colpivot, A)
```

In parallel
```
using MPISharedMemLUs
using LinearAlgebra
using MPI

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

n = 4

if rank == 0
    dims_local = (n, n)
else
    dims_local = (0, 0)
end
winA, A_temp = MPI.Win_allocate_shared(Array{Int64}, dims_local, comm)
A = MPI.Win_shared_query(Array{Int64}, (n, n), win; rank=0)

if rank == 0
    dims_local = (n,)
else
    dims_local = (0,)
end
winpiv, piv_temp = MPI.Win_allocate_shared(Array{Int64}, dims_local, comm)
piv = MPI.Win_shared_query(Array{Int64}, (n,), winpiv; rank=0)

if rank == 0
    A .= rand(n, n)
end
MPI.Barrier(comm)

Alu_row_pivot = get_row_pivot_lu(piv, comm)

lu!(Alu_rowpivot, A)

MPI.Barrier(comm)

if rank == 0
    A .= rand(n, n)
end
MPI.Barrier(comm)

Alu_col_pivot = get_column_pivot_lu(piv, comm)

lu!(Alu_colpivot, A)

MPI.Free(win)
MPI.Free(winpiv)
```

After the `lu!()` calls, the LU factorized matrix is contained in `A`, and the
row-pivot indices in `Alu_row_pivot.ipiv`, or column-pivot indi in
`Alu_col_pivot.jpiv`.
