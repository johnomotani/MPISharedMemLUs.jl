using BenchmarkTools
using LinearAlgebra
using ColumnPivotLUs

function serial_benchmark(short_size::Integer, long_size::Integer)
    # Here we compare serial version of all algorithms.
    BLAS.set_num_threads(1)

    Ac = rand(short_size,long_size)
    jpiv = zeros(Int64,short_size)
    Ar = Matrix(transpose(Ac))
    ipiv = zeros(Int64,short_size)

    println("Benchmark short_size=$short_size, long_size=$long_size")
    println("===========================================")
    println()

    println("LAPACK")
    b = @benchmark LAPACK.getrf!(Acopy, $ipiv; check=false) setup=(Acopy=copy($Ar))
    display(b)

    println("\nRow pivoting")
    rplu = get_row_pivot_lu(ipiv)
    b = @benchmark lu!($rplu, Acopy) setup=(Acopy=copy($Ar))
    display(b)

    println("\ncolumn pivoting")
    cplu = get_column_pivot_lu(jpiv)
    b = @benchmark lu!($cplu, Acopy) setup=(Acopy=copy($Ac))
    display(b)

    println()

    return nothing
end

serial_benchmark(128, 4096)
serial_benchmark(4096, 4096)
serial_benchmark(8192, 8192)
