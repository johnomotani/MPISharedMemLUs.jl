using BenchmarkTools
using Dates
using LinearAlgebra
using MPI
using StableRNGs
using ColumnPivotLUs

function mpi_benchmark(short_size::Integer, long_size::Integer, nsamples::Integer)
    # Here we compare serial version of all algorithms.
    BLAS.set_num_threads(1)

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    local_win_store_float = nothing
    local_win_store_float = MPI.Win[]
    allocate_shared_float = (dims...)->begin
        if rank == 0
            dims_local = dims
        else
            dims_local = Tuple(0 for _ ∈ dims)
        end
        win, array_temp = MPI.Win_allocate_shared(Array{Float64}, dims_local, comm)
        array = MPI.Win_shared_query(Array{Float64}, dims, win; rank=0)
        push!(local_win_store_float, win)
        if rank == 0
            array .= NaN
        end
        MPI.Barrier(comm)
        return array
    end

    local_win_store_int = MPI.Win[]
    allocate_shared_int = (dims...)->begin
        if rank == 0
            dims_local = dims
        else
            dims_local = Tuple(0 for _ ∈ dims)
        end
        win, array_temp = MPI.Win_allocate_shared(Array{Int64}, dims_local, comm)
        array = MPI.Win_shared_query(Array{Int64}, dims, win; rank=0)
        push!(local_win_store_int, win)
        if rank == 0
            array .= typemin(Int64)
        end
        MPI.Barrier(comm)
        return array
    end

    Acopy = allocate_shared_float(short_size, long_size)
    index_buffer = allocate_shared_int(short_size)
    maxabs_buffer = allocate_shared_float(short_size)
    if rank == 0
        rng = StableRNG(42)
        Ac = rand(short_size, long_size)
        jpiv = zeros(Int64, short_size)

        println("ColumnPivotLUs Benchmark np=$nproc short_size=$short_size, long_size=$long_size, ", now())
        println("=====================================================================================")
        println()
    else
        Ac = nothing
        jpiv = zeros(Int64, 0)
    end

    MPI.Barrier(comm)

    # Do a fixed number of samples (with no limit on benchmark time) to ensure that all
    # MPI processes do exactly the same number of calls. Otherwise, the benchmark will
    # hang intermittently.
    b = @benchmark column_pivot_lu!($Acopy, $jpiv, $comm, $index_buffer, $maxabs_buffer) setup=($rank == 0 && copyto!($Acopy, $Ac); MPI.Barrier($comm)) teardown=(MPI.Barrier($comm)) samples=nsamples evals=1 seconds=Inf

    if rank == 0
        display(b)
        println()
    end

    # Free the MPI.Win objects, because if they are free'd by the garbage collector it may
    # cause an MPI error or hang.
    for w ∈ local_win_store_float
        MPI.free(w)
    end
    resize!(local_win_store_float, 0)
    for w ∈ local_win_store_int
        MPI.free(w)
    end
    resize!(local_win_store_int, 0)

    return nothing
end

MPI.Init()
mpi_benchmark(128, 4096, 4000)
mpi_benchmark(4096, 4096, 20)
mpi_benchmark(8192, 8192, 4)
