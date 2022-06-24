module QubitSystems

export AbstractQubitSystem
export SingleQubitSystem
export dynamics
export objective

using ..QuantumLogic
using ..Dynamics

using LinearAlgebra

abstract type AbstractQubitSystem{N} end

struct SingleQubitSystem <: AbstractQubitSystem{1}
    nvars::Int
    nstates::Int
    nqstates::Int
    isodim::Int
    control_order::Int
    T::Int
    Δt::Float64
    integrator::Function
    loss::Function
    ψ̃1::Vector{Float64}
end

function SingleQubitSystem(
    H_drift::Matrix,
    H_drive::Matrix,
    gate::Symbol,
    ψ1::Union{Vector{C}, Vector{Vector{C}}};
    control_order=2,
    T=1000,
    Δt=0.01,
    integrator=pade_schroedinger,
    loss=amplitude_loss,
) where C <: Number

    if isa(ψ1, Vector{C})
        nqstates = 1
        isodim = 2 * length(ψ1)
        ψ̃goal = apply(gate, ψ1) |> ket_to_iso
        my_loss = ψ̃ -> loss(ψ̃, ψ̃goal)
        ψ̃1 = ket_to_iso(ψ1)
    else
        nqstates = length(ψ1)
        isodim = 2 * length(ψ1[1])
        ψ̃goal = apply.(gate, ψ1) .|> ket_to_iso
        my_loss = ψ̃ -> sum(
            [loss(ψ̃[(1 + (i - 1)*isodim):i*isodim], ψ̃goal[i]) for i = 1:nqstates]
        )
        ψ̃1 = vcat(ket_to_iso.(ψ1)...)
    end

    Im2 = [0 -1; 1 0]

    ⊗(A, B) = kron(A, B)

    G(H) = I(2) ⊗ imag(H) - Im2 ⊗ real(H)

    G_drift = G(H_drift)
    G_drive = G(H_drive)

    my_integrator = (ψ̃ₜ₊₁, ψ̃ₜ, aₜ) -> integrator(ψ̃ₜ₊₁, ψ̃ₜ, aₜ, Δt, G_drift, G_drive)

    nstates = nqstates * isodim + control_order + 2

    return SingleQubitSystem(
        nstates*T,
        nstates,
        nqstates,
        isodim,
        control_order,
        T,
        Δt,
        my_integrator,
        my_loss,
        ψ̃1
    )
end

function dynamics(
    system::AbstractQubitSystem{N},
    xₜ₊₁,
    xₜ,
    uₜ
) where {N,T}
    Δt = system.Δt
    aₜ = xₜ[end-system.control_order+1]
    aₜs = xₜ[(end-system.control_order):end]
    aₜ₊₁s = xₜ₊₁[(end-system.control_order):end]
    âₜ₊₁s = zeros(typeof(xₜ[1]), system.control_order + 1)
    for i = 1:system.control_order
        âₜ₊₁s[i] = Δt * aₜs[i+1]
    end
    âₜ₊₁s[end] = Δt * uₜ
    δaₜ₊₁s = aₜ₊₁s - âₜ₊₁s
    ψ̃ₜs = [xₜ[(1+(i-1)*system.isodim):i*system.isodim] for i = 1:system.nqstates]
    ψ̃ₜ₊₁s = [xₜ₊₁[(1+(i-1)*system.isodim):i*system.isodim] for i = 1:system.nqstates]
    δψ̃ₜ₊₁s = [system.integrator(ψ̃ₜ₊₁s[i], ψ̃ₜs[i], aₜ) for i = 1:system.nqstates]
    δxₜ₊₁ = vcat(δψ̃ₜ₊₁s..., δaₜ₊₁s)
    return δxₜ₊₁
end

function objective(
    system::AbstractQubitSystem{N},
    xs,
    u,
    Q,
    Qf,
    R
) where {N,T}
    ψ̃ts = [xs[t][1:(system.isodim*system.nqstates)] for t = 1:system.T]
    cost = dot([fill(Q, system.T-1); Qf], system.loss.(ψ̃ts))
    cost += R * dot(u, u)
    return cost
end


end
