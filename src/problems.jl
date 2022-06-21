module Problems

export MultiQubitProblem
export SingleQubitProblem
export qubit_num
export solve

using ..QuantumLogic
using ..Dynamics

using LinearAlgebra
using JuMP
import Ipopt

abstract type MultiQubitProblem{N} end

struct SingleQubitProblem <: MultiQubitProblem{1}
    G_drift
    G_drive
    nqstates::Int
    isodim::Int
    control_order::Int
    T::Int
    Δt::Float64
    loss::Function
    ψ̃1::Vector{Float64}
end

function SingleQubitProblem(
    H_drift::Matrix,
    H_drive::Matrix,
    gate::Symbol,
    ψ1::Union{Vector{C}, Vector{Vector{C}}};
    control_order=2,
    T=1000,
    Δt=0.01,
    state_loss=amplitude_loss
) where C <: Number

    if isa(ψ1, Vector{C})
        nqstates = 1
        isodim = 2 * length(ψ1)
        ψ̃goal = apply(gate, ψ1) |> ket_to_iso
        loss = ψ̃ -> state_loss(ψ̃, ψ̃goal)
        ψ̃1 = ket_to_iso(ψ1)
    else
        nqstates = length(ψ1)
        isodim = 2 * length(ψ1[1])
        ψ̃goal = apply.(gate, ψ1) .|> ket_to_iso
        loss = ψ̃ -> sum(
            [loss(ψ̃[(1 + (i - 1)*isodim):i*isodim], ψ̃goal[i]) for i = 1:nqstates]
        )
        ψ̃1 = foldr(vcat, ket_to_iso.(ψ1))
    end

    Im2 = [0 -1; 1 0]

    ⊗(A, B) = kron(A, B)

    G(H) = I(2) ⊗ imag(H) - Im2 ⊗ real(H)

    G_drift = G(H_drift)
    G_drive = G(H_drive)

    return SingleQubitProblem(
        G_drift,
        G_drive,
        nqstates,
        isodim,
        control_order,
        T,
        Δt,
        loss,
        ψ̃1
    )
end

function solve(
    prob::MultiQubitProblem{N};
    dynamics=pade_schroedinger,
    ubound=0.5,
    Q=0.1,
    Qf=100,
    R=0.5
) where N

    model = JuMP.Model(Ipopt.Optimizer)

    # create state variable: X[:, t] = [ψ₁, ..., ψₙ, ∫a, a, da, ..., d⁽ᵒʳᵈᵉʳ⁻¹⁾a ]
    #               default: X[:, t] = [ψ₁, ..., ψₙ, ∫a, a, da]
    @variable(model, X[1:(prob.nqstates * prob.isodim + prob.control_order - 1), 1:prob.T])

    # create bounded control variable u: u[t] = dda (default)
    @variable(model, -ubound <= u[1:(prob.T - 1)] <= ubound)

    # create objective function
    @NLobjective(
        model,
        Min,
        Q * sum(prob.loss.([X[1:(prob.isodim*prob.nqstates), t] for t = 1:prob.T-1])) +
        Qf * prob.loss(X[1:(prob.isodim*prob.nqstates), end]) +
        R * sum(u[t]^2 for t = 1:prob.T-1)
    )


    # add dynamics constraints
    for t = 1:prob.T-1
        for i = 1:prob.nqstates
            @NLconstraint(
                model,
                dynamics(
                    X[(1 + (i - 1)*prob.isodim):(i*prob.isodim), t + 1],
                    X[(1 + (i - 1)*prob.isodim):(i*prob.isodim), t],
                    u[t],
                    prob.Δt,
                    prob.G_drift,
                    prob.G_drive
                ) == 0
            )
        end
    end

    @show model
end

function objective(prob::MultiQubitProblem, X, u, Q, Qf, R)
    ψ̃ts = [X[1:(prob.isodim*prob.nqstates), t] for t = 1:prob.T]
    J = dot([fill(Q, prob.T-1); Qf], prob.loss.(ψ̃ts))
    J += R * dot(u, u)
    return J
end


end
