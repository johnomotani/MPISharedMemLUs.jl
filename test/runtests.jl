using ColumnPivotLUs
using LinearAlgebra
using MPI
using StableRNGs
using Test

function get_comms()
    comm = MPI.COMM_WORLD
    nproc = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

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

    return comm, rank, nproc, allocate_shared_float, allocate_shared_int,
           local_win_store_float, local_win_store_int
end

function test_column_pivoting(m, n, tol)
    rng = StableRNG(234)
    @testset "column pivoting" begin
        A = rand(rng, m, n)
        s = min(m, n)
        jpiv = zeros(Int64, s)
        ALU = copy(A)
        column_pivot_lu!(ALU, jpiv)
        L = fill(NaN, m, s)
        L[1:s,1:s] .= UnitLowerTriangular(ALU[1:s,1:s])
        L[s+1:end,1:s] .= ALU[s+1:end,1:s]
        U = fill(NaN, s, n)
        U[1:s,1:s] .= UpperTriangular(ALU[1:s,1:s])
        U[1:s,s+1:end] .= ALU[1:s,s+1:end]
        p = LinearAlgebra.ipiv2perm(jpiv, n)
        @test isapprox(L * U, A[:,p], atol=tol, norm=x->NaN)
    end
    return nothing
end

function test_column_pivoting_mpi(m, n, tol)
    rng = StableRNG(234)
    comm, rank, nproc, allocate_shared_float, allocate_shared_int, local_win_store_float,
        local_win_store_int = get_comms()

    @testset "column pivoting with mpi" begin
        ALU = allocate_shared_float(m, n)
        index_buffer = allocate_shared_int(nproc)
        maxabs_buffer = allocate_shared_float(nproc)
        if rank == 0
            A = rand(rng, m, n)
            ALU .= A
            index_buffer .= -1
            maxabs_buffer .= NaN
        end
        MPI.Barrier(comm)
        s = min(m, n)
        jpiv = zeros(Int64, s)
        column_pivot_lu!(ALU, jpiv, comm, index_buffer, maxabs_buffer)
        if rank == 0
            L = fill(NaN, m, s)
            L[1:s,1:s] .= UnitLowerTriangular(ALU[1:s,1:s])
            L[s+1:end,1:s] .= ALU[s+1:end,1:s]
            U = fill(NaN, s, n)
            U[1:s,1:s] .= UpperTriangular(ALU[1:s,1:s])
            U[1:s,s+1:end] .= ALU[1:s,s+1:end]
            p = LinearAlgebra.ipiv2perm(jpiv, n)
            @test isapprox(L * U, A[:,p], atol=tol, norm=x->NaN)
        end
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

function test_row_pivoting(m, n, tol)
    rng = StableRNG(123)
    @testset "row pivoting" begin
        A = rand(rng, m, n)
        s = min(m, n)
        ipiv = zeros(Int64, s);
        ALU = copy(A)
        row_pivot_lu!(ALU, ipiv)
        L = fill(NaN, m, s)
        L[1:s,1:s] .= UnitLowerTriangular(ALU[1:s,1:s])
        L[s+1:end,1:s] .= ALU[s+1:end,1:s]
        U = fill(NaN, s, n)
        U[1:s,1:s] .= UpperTriangular(ALU[1:s,1:s])
        U[1:s,s+1:end] .= ALU[1:s,s+1:end]
        p = LinearAlgebra.ipiv2perm(ipiv, m)
        @test isapprox(L * U, A[p,:], atol=tol, norm=x->NaN)
    end
    return nothing
end

@testset "ColumnPivotLUs.jl" begin
    if !MPI.Initialized()
        MPI.Init()
    end
    BLAS.set_num_threads(1)
    tol = 4.0e-13
    @testset "m=$m n=$n" for m ∈ [16, 32, 53, 64, 128, 143, 4096], n ∈ [16, 32, 53, 64, 128, 143, 4096]
        test_column_pivoting(m, n, tol)
        test_column_pivoting_mpi(m, n, tol)
        test_row_pivoting(m, n, tol)
    end
end
