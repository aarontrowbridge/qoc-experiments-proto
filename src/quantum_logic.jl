module QuantumLogic

export GATES
export ⊗
export apply
export ket_to_iso
export iso_to_ket
export annihilate
export create
export quad
export cavity_state
export number
export normalize
export get_mat_iso
export get_mat_uniso
export basis

using LinearAlgebra

const GATES = Dict(
    :X => [0 1;
           1 0],

    :Y => [0 -im;
           im 0],

    :Z => [1 0;
           0 -1],

    :H => [1 1;
           1 -1]/√2,

    :CX => [1 0 0 0;
            0 1 0 0;
            0 0 0 1;
            0 0 1 0],

    :XI => [0 0 -im 0;
            0 0 0 -im;
            -im 0 0 0;
            0 -im 0 0],

    :sqrtiSWAP => [1 0 0 0;
                   0 1/sqrt(2) 1im/sqrt(2) 0;
                   0 1im/sqrt(2) 1/sqrt(2) 0;
                   0 0 0 1]
)

⊗(A, B) = kron(A, B)

function apply(gate::Symbol, ψ::Vector{T} where T<:Number)
    @assert norm(ψ) ≈ 1.0
    @assert gate in keys(GATES) "gate not found"
    U = GATES[gate]
    @assert size(U)[2] == size(ψ)[1] "gate size does not match ket dim"
    return ComplexF64.(normalize(U * ψ))
end


"""
    quantum harmonic oscillator operators

"""

function annihilate(levels::Int)
    return diagm(1 => map(sqrt, 1:levels - 1))
end


function create(levels::Int)
    return (annihilate(levels))'
end

function number(levels::Int)
    return create(levels) * annihilate(levels)
end

function quad(levels::Int)
    return number(levels)*(number(levels) - I(levels))
end

function cavity_state(level, cavity_levels)
    state = zeros(cavity_levels)
    state[level + 1] = 1.
    return state
end

ket_to_iso(ψ) = [real(ψ); imag(ψ)]

iso_to_ket(ψ̃) =
    ψ̃[1:div(length(ψ̃), 2)] +
    im * ψ̃[(div(length(ψ̃), 2) + 1):end]

function normalize(state::Vector{C} where C <: Number)
    return state / norm(state)
end

function get_mat_iso(mat)
    mat_r = real(mat)
    mat_i = imag(mat)
    return vcat(hcat(mat_r, -mat_i),
                hcat(mat_i,  mat_r))
end

function get_mat_uniso(mat)
    n = size(mat, 1)
    nby2 = Integer(n/2)
    i1 = 1:nby2
    i2 = (nby2 + 1):n
    return mat[i1, i1] + mat[i2, i1]
end

function basis(index::Int, dim::Int)
    if index > dim || index < 1
        throw(DomainError(index, "Index out of bounds"))
    end
    [zeros(index - 1); 1; zeros(dim - index)]
end

end
