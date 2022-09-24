module InterativeLearningControl

export QuantumILCProblem
export MeasurementData
export measure

using ..Trajectories
using ..QuantumSystems
using ..Dynamics
using ..Integrators
using ..Utils

using LinearAlgebra
using SparseArrays
using ForwardDiff
using Einsum
using OSQP

struct MeasurementData
    ys::Vector{Vector{Float64}}
    times::AbstractVector{Int}
    ydim::Int
end

function measure(
    Z::Trajectory,
    g::Function,
    times::AbstractVector{Int},
    ydim::Int,
)
    @assert size(g(Z.states[1]), 1) == ydim
    ys = Vector{Vector{Float64}}(undef, length(times))
    for (i, t) in enumerate(times)
        ys[i] = g(Z.states[t])
    end
    return MeasurementData(ys, times, ydim)
end

abstract type ILCProblem end

struct QuantumILCProblem <: ILCProblem
    system::QuantumSystem
    Ẑ::Trajectory
    g::Function
    Ŷ::MeasurementData
    Ȳ::MeasurementData
    H::SparseMatrixCSC{Float64, Int}
    A::SparseMatrixCSC{Float64, Int}
    dims::NamedTuple
end


function QuantumILCProblem(
    sys::QuantumSystem,
    Ẑ::Trajectory,
    g::Function,
    Ȳ::MeasurementData;
    Q=1.0,
    R=1.0,
    integrator=:FourthOrderPade
)
    dims = (
        z = size(Ẑ.states[1], 1) + size(Ẑ.actions[1], 1),
        x = size(Ẑ.states[1], 1),
        f = size(Ẑ.states[1], 1),
        u = size(Ẑ.actions[1], 1),
        y = Ȳ.ydim,
        T = Ẑ.T,
        M = length(Ȳ.times)
    )

    Ŷ = measure(Ẑ, g, Ȳ.times, dims.y)

    f = eval(integrator)(sys)

    A = build_constraint_matrix(f, g, Ẑ, Ŷ, Ȳ, dims)

    H = build_hessian(Q, R, dims)

    return QuantumILCProblem(sys, Ẑ, g, Ŷ, Ȳ, H, A, dims)
end

function build_hessian(
    Q::Float64,
    R::Float64,
    dims::NamedTuple
)
    Hₜ = spdiagm([Q * ones(dims.x); R * ones(dims.u)])
    H = kron(sparse(I(dims.T)), Hₜ)
    return H
end

function build_constraint_matrix(
    f::AbstractQuantumIntegrator,
    g::Function,
    Ẑ::Trajectory,
    Ŷ::MeasurementData,
    Ȳ::MeasurementData,
    dims::NamedTuple;
    correction_term=true
)
    ∇F = build_dynamics_constraint_jacobian(f, Ẑ, dims)

    ∇G = build_measurement_constraint_jacobian(
        g, Ŷ, Ȳ, dims;
        correction_term=correction_term
    )

    A = sparse_vcat(∇F, ∇G)

    return A
end

function build_measurement_constraint_jacobian(
    g::Function,
    Ŷ::MeasurementData,
    Ȳ::MeasurementData,
    dims::NamedTuple;
    correction_term=true
)
    ∇g(x) = ForwardDiff.jacobian(g, x)

    function ∇²g(x)
        H = ForwardDiff.jacobian(u -> vec(∇g(u)), x)
        return reshape(H, dims.y, dims.x, dims.x)
    end

    ∇G = spzeros(dims.y * dims.M, dims.z * dims.T)

    Δys = Ȳ.ys .- Ŷ.ys

    for (i, (τᵢ, Δyᵢ)) in enumerate(zip(Ȳ.times, Δys))

        ∇gᵢ = ∇g(Ẑ.states[τᵢ])

        if correction_term
            ∇²gᵢ = ∇²g(Ẑ.states[τᵢ])
            ϵ̂ᵢ = pinv(∇gᵢ) * Δyᵢ
            @einsum ∇gᵢ[j, k] += ∇²gᵢ[j, k, l] * ϵ̂ᵢ[l]
        end

        ∇G[
            slice(i, dims.y),
            slice(τᵢ, dims.x, dims.z)
        ] = sparse(∇gᵢ)
    end

    return ∇G
end

function build_dynamics_constraint_jacobian(
    f::AbstractQuantumIntegrator,
    X̂::Trajectory,
    dims::NamedTuple
)
    ∇f = Jacobian(f)

    ∇F = spzeros(dims.x * (dims.T - 1), dims.z * dims.T)

    for t = 1:dims.T - 1
        ∇fₜ = spzeroes(dims.x, 2 * dims.z)

        xₜ = X̂.states[t]
        uₜ = X̂.actions[t]
        xₜ₊₁ = X̂.states[t + 1]

        ∂xₜfₜ = ∇f(uₜ, X̂.Δt, false)

        ∇fₜ[1:dims.x, 1:dims.x] = sparse(∂xₜfₜ)

        ∂uʲₜfₜs = [∇f(xₜ₊₁, xₜ, uₜ, X̂.Δt, j) for j = 1:dims.u]

        for (j, ∂uʲₜfₜ) in enumerate(∂uʲₜfₜs)
            ∇fₜ[1:dims.x, dims.x + j] = sparse(∂uʲₜfₜ)
        end

        ∂xₜ₊₁fₜ = ∇f(uₜ, X̂.Δt, true)

        ∇fₜ[1:dims.x, dims.z .+ (1:dims.x)] = ∂xₜ₊₁fₜ

        ∇F[
            slice(t, dims.x),
            slice(t, dims.z; stretch=dims.z)
        ] = ∇fₜ
    end

    return ∇F
end


end
