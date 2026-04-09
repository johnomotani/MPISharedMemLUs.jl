module ColumnPivotLUs

export row_pivot_lu!, column_pivot_lu!

using LoopVectorization
using LinearAlgebra
using LinearAlgebra.BLAS: trsm!

function find_pivot(a, n)
    pivot_ind = 1
    maxabs = abs(a[1])
    for j ∈ 2:n
        thisabs = abs(a[j])
        if thisabs > maxabs
            maxabs = thisabs
            pivot_ind = j
        end
    end
    return pivot_ind
end

function apply_column_swaps!(A, jpiv, m, npivot)
    for j ∈ 1:npivot
        pivot_ind = jpiv[j]
        for i ∈ 1:m
            A[i,j], A[i,pivot_ind] = A[i,pivot_ind], A[i,j]
        end
    end
    return nothing
end

function column_pivot_lu!(A::AbstractMatrix, jpiv::AbstractVector{<:Integer})
    recursive_column_pivot_lu!(A, jpiv, size(A, 1), size(A, 2))
    return A
end

function recursive_column_pivot_lu!(A::AbstractMatrix, jpiv::AbstractVector{<:Integer},
                                    m::Integer, n::Integer)
    # A - the matrix being factorised in-place.
    # jpiv - the (column) pivot indices.
    # m - the number of rows in A.
    # n - the number of columns in A.

    # This function borrows heavily from DGETRF2 from LAPACK, v3.12.1.
    # Recurse not over rows/columns but by splitting the matrix approximately in half each
    # step.

    @inbounds begin
        # Quick return if possible.
        if m == 0 || n == 0
            return nothing
        end

        if n == 1
            # One column case, just need to handle jpiv and update column.
            jpiv[1] = 1
            @views A[2:end,1] .*= 1.0 / A[1,1]
        elseif m == 1
            # One row case.
            pivot_ind = find_pivot(@view(A[1,:]), n)
            jpiv[1] = pivot_ind

            # Apply the interchange
            A[1,1], A[1,pivot_ind] = A[1,pivot_ind], A[1,1]
        else
            # Block-factorise
            # [ A11 | A12 ]
            # [ --------- ]
            # [ A21 | A22 ]
            m1 = min(m, n) ÷ 2
            m2 = m - m1
            n2 = n - m1

            # Factor
            # [ A11 | A12 ]
            recursive_column_pivot_lu!(@view(A[1:m1,:]), jpiv, m1, n)

            # Apply interchanges to
            # [ A21 | A22 ]
            apply_column_swaps!(@view(A[m1+1:m,:]), jpiv, m2, m1)

            # Solve A21
            A21 = @view A[m1+1:m,1:m1]
            @views trsm!('R', 'U', 'N', 'N', 1.0, A[1:m1,1:m1], A21)
            #A11 = @view A[1:m1,1:m1]
            #for i ∈ 1:m2, j ∈ 1:m1
            #    for k ∈ 1:j-1
            #        A21[i,j] -= A11[k,j] * A21[i,k]
            #    end
            #    A21[i,j] /= A11[j,j]
            #end

            # Update A22
            A12 = @view A[1:m1,m1+1:n]
            A22 = @view A[m1+1:m,m1+1:n]
            #mul!(A22, A21, A12, -1.0, 1.0)
            @turbo for j ∈ 1:n2, k ∈ 1:m1, i ∈ 1:m2
                A22[i,j] -= A21[i,k] * A12[k,j]
            end

            # Factor A22
            right_jpiv = @view jpiv[m1+1:min(m,n)]
            recursive_column_pivot_lu!(A22, right_jpiv, m2, n2)

            # Apply interchanges to A12
            apply_column_swaps!(A12, right_jpiv, m1, min(m2,n2))

            right_jpiv .+= m1
        end
    end
    return nothing
end

function apply_row_swaps!(A, ipiv, n, mpivot)
    for i ∈ 1:mpivot
        pivot_ind = ipiv[i]
        for j ∈ 1:n
            A[i,j], A[pivot_ind,j] = A[pivot_ind,j], A[i,j]
        end
    end
    return nothing
end

function row_pivot_lu!(A::AbstractMatrix, ipiv::AbstractVector{<:Integer})
    recursive_row_pivot_lu!(A, ipiv, size(A, 1), size(A, 2))
    return A
end

function recursive_row_pivot_lu!(A::AbstractMatrix, ipiv::AbstractVector{<:Integer},
                                 m::Integer, n::Integer)
    # A - the matrix being factorised in-place.
    # ipiv - the (row) pivot indices.
    # m - the number of rows in A.
    # n - the number of columns in A.

    # This function is essentially a copy of DGETRF2 from LAPACK, v3.12.1.
    # Recurse not over rows/columns but by splitting the matrix approximately in half each
    # step.

    @inbounds begin
        # Quick return if possible.
        if m == 0 || n == 0
            return nothing
        end

        if m == 1
            # One row case, just need to handle ipiv.
            ipiv[1] = 1
        elseif n == 1
            # One column case.
            pivot_ind = find_pivot(@view(A[:,1]), m)
            ipiv[1] = pivot_ind

            # Apply the interchange
            A[1,1], A[pivot_ind,1] = A[pivot_ind,1], A[1,1]

            # Update the column
            @views A[2:end,1] .*= 1.0 / A[1,1]
        else
            # Block-factorise
            # [ A11 | A12 ]
            # [ --------- ]
            # [ A21 | A22 ]
            n1 = min(m, n) ÷ 2
            n2 = n - n1
            m2 = m - n1

            # Factor
            # [ A11 ]
            # [ --- ]
            # [ A21 ]
            recursive_row_pivot_lu!(@view(A[:,1:n1]), ipiv, m, n1)

            # Apply interchanges to
            # [ A12 ]
            # [ --- ]
            # [ A22 ]
            apply_row_swaps!(@view(A[:,n1+1:n]), ipiv, n2, n1)

            # Solve A12
            A12 = @view A[1:n1,n1+1:n]
            @views trsm!('L', 'L', 'N', 'U', 1.0, A[1:n1,1:n1], A12)
            #A11 = @view A[1:n1,1:n1]
            #for j ∈ 1:n2, i ∈ 1:n1-1
            #    for k ∈ i+1:n1
            #        A12[k,j] -= A11[k,i] * A12[i,j]
            #    end
            #end

            # Update A22
            A21 = @view A[n1+1:m,1:n1]
            A22 = @view A[n1+1:m,n1+1:n]
            #mul!(A22, A21, A12, -1.0, 1.0)
            @turbo for j ∈ 1:n2, k ∈ 1:n1, i ∈ 1:m2
                A22[i,j] -= A21[i,k] * A12[k,j]
            end

            # Factor A22
            bottom_ipiv = @view ipiv[n1+1:min(m,n)]
            recursive_row_pivot_lu!(A22, bottom_ipiv, m2, n2)

            # Apply interchanges to A21
            apply_row_swaps!(A21, bottom_ipiv, n1, min(m2,n2))

            bottom_ipiv .+= n1
        end
    end
    return nothing
end

end
