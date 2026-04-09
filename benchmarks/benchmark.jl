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

    println("LAPACK")
    b = @benchmark LAPACK.getrf!(Acopy, $ipiv) setup=(Acopy=copy($Ar))
    display(b)

    println("\nRow pivoting")
    b = @benchmark row_pivot_lu!(Acopy, $ipiv) setup=(Acopy=copy($Ar))
    display(b)

    println("\ncolumn pivoting")
    b = @benchmark column_pivot_lu!(Acopy, $jpiv) setup=(Acopy=copy($Ac))
    display(b)

    return nothing
end

serial_benchmark(128, 4096)
