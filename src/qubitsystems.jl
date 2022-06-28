module QubitSystems

export AbstractQubitSystem
export SingleQubitSystem

using ..QuantumLogic

using LinearAlgebra

abstract type AbstractQubitSystem{N} end

struct SingleQubitSystem <: AbstractQubitSystem{1}
    nstates::Int
    nqstates::Int
    isodim::Int
    control_order::Int
    G_drift::Matrix{Float64}
    G_drive::Matrix{Float64}
    ψ̃1::Vector{Float64}
    ψ̃goal::Vector{Float64}
end

function SingleQubitSystem(
    H_drift::Matrix,
    H_drive::Matrix,
    gate::Symbol,
    ψ1::Union{Vector{C}, Vector{Vector{C}}};
    control_order=2
) where C <: Number

    if isa(ψ1, Vector{C})
        nqstates = 1
        isodim = 2 * length(ψ1)
        ψ̃goal = ket_to_iso(apply(gate, ψ1))
        ψ̃1 = ket_to_iso(ψ1)
    else
        nqstates = length(ψ1)
        isodim = 2 * length(ψ1[1])
        ψ̃goal = vcat(ket_to_iso.(apply.(gate, ψ1))...)
        ψ̃1 = vcat(ket_to_iso.(ψ1)...)
    end

    Im2 = [0 -1; 1 0]

    ⊗(A, B) = kron(A, B)

    G(H) = I(2) ⊗ imag(H) - Im2 ⊗ real(H)

    G_drift = G(H_drift)
    G_drive = G(H_drive)

    nstates = nqstates * isodim + control_order + 1

    return SingleQubitSystem(
        nstates,
        nqstates,
        isodim,
        control_order,
        G_drift,
        G_drive,
        ψ̃1,
        ψ̃goal,
    )
end


end
