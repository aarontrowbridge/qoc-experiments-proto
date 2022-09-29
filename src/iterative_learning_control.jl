module IterativeLearningControl

export ILCProblem
export solve!

export QuadraticProblem

export MeasurementData
export measure

export QuantumExperiment
export experiment

using ..Trajectories
using ..QuantumSystems
using ..Integrators
using ..Utils
using ..ProblemSolvers
using ..QuantumLogic

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

function Base.:-(
    data1::MeasurementData,
    data2::MeasurementData
)
    @assert data1.ydim == data2.ydim
    @assert data1.times == data2.times
    return MeasurementData(
        data1.ys .- data2.ys,
        data1.times,
        data1.ydim
    )
end

function LinearAlgebra.norm(data::MeasurementData, p::Real=2)
    return norm(vcat(data.ys...), p)
end

function measure(
    Z::Trajectory,
    g::Function,
    τs::AbstractVector{Int},
    ydim::Int,
)
    @assert size(g(Z.states[1]), 1) == ydim
    ys = Vector{Vector{Float64}}(undef, length(τs))
    for (i, τ) in enumerate(τs)
        ys[i] = g(Z.states[τ])
    end
    return MeasurementData(ys, τs, ydim)
end

function measure(
    Ψ̃::Vector{Vector{Float64}},
    g::Function,
    τs::AbstractVector{Int},
    ydim::Int,
)
    @assert size(g(Ψ̃[1]), 1) == ydim
    ys = Vector{Vector{Float64}}(undef, length(τs))
    for (i, τ) in enumerate(τs)
        ys[i] = g(Ψ̃[τ])
    end
    return MeasurementData(ys, τs, ydim)
end

struct QuantumExperiment
    sys::QuantumSystem
    ψ̃₁::Vector{Float64}
    Δt::Float64
    g::Function
    ydim::Int
    τs::AbstractVector{Int}
    integrator::Function
end

function QuantumExperiment(
    sys::QuantumSystem,
    ψ̃₁::Vector{Float64},
    Δt::Float64,
    g::Function,
    ydim::Int,
    τs::AbstractVector{Int};
    integrator=exp
)
    return QuantumExperiment(
        sys,
        ψ̃₁,
        Δt,
        g,
        ydim,
        τs,
        integrator
    )
end

function fourth_order_pade(Gₜ::Matrix)
    Id = I(size(Gₜ, 1))
    return inv(Id - Δt / 2 * Gₜ + Δt^2 / 9 * Gₜ^2) *
        (Id + Δt / 2 * Gₜ + Δt^2 / 9 * Gₜ^2)
end

function second_order_pade(Gₜ::Matrix)
    Id = I(size(Gₜ, 1))
    return inv(Id - Δt / 2 * Gₜ) *
        (Id + Δt / 2 * Gₜ)
end


# TODO:
# - add noise terms (must correspond to ketdim)
# - add multiple quantum state functionality here
# - show fidelity

function (experiment::QuantumExperiment)(
    U::Vector{Vector{Float64}}
)::MeasurementData

    T = length(U)

    Ψ̃ = Vector{typeof(experiment.ψ̃₁)}(undef, T)
    Ψ̃[1] = experiment.ψ̃₁

    for t = 2:T
        Gₜ = Integrators.G(
            U[t - 1],
            experiment.sys.G_drift,
            experiment.sys.G_drives
        ) + 0.5e-2 * QuantumSystems.G(GATES[:CX])
        Ψ̃[t] = experiment.integrator(Gₜ * experiment.Δt) *
            Ψ̃[t - 1]
    end

    Ȳ = measure(
        Ψ̃,
        experiment.g,
        experiment.τs,
        experiment.ydim
    )

    return Ȳ
end


# TODO:
# - make more general for arbitrary dynamics
#   - make work with augmented state
# - add functionality to store non updating jacobians
# - OSQP error handling

struct QuadraticProblem
    f::AbstractIntegrator
    ∇g::Function
    ∇²g::Function
    Q::Float64
    R::Float64
    u_bounds::Vector{Float64}
    correction_term::Bool
    settings::Dict
    dims::NamedTuple
end

function QuadraticProblem(
    f::AbstractIntegrator,
    g::Function,
    Q::Float64,
    R::Float64,
    u_bounds::Vector{Float64},
    correction_term::Bool,
    settings::Dict,
    xdim::Int,
    udim::Int,
    ydim::Int,
    T::Int,
    M::Int,
)
    dims = (
        z=xdim + udim,
        x=xdim,
        u=udim,
        y=ydim,
        T=T,
        M=M
    )

    ∇g(x) = ForwardDiff.jacobian(g, x)

    function ∇²g(x)
        H = ForwardDiff.jacobian(u -> vec(∇g(u)), x)
        return reshape(H, dims.y, dims.x, dims.x)
    end

    return QuadraticProblem(
        f,
        ∇g,
        ∇²g,
        Q,
        R,
        u_bounds,
        correction_term,
        settings,
        dims
    )
end

function (QP::QuadraticProblem)(
    Ẑ::Trajectory,
    ΔY::MeasurementData,
)
    H = build_hessian(QP.Q, QP.R, QP.dims)

    A, lb, ub = build_constraint_matrix(
        QP,
        Ẑ,
        ΔY
    )

    model = OSQP.Model()

    OSQP.setup!(
        model;
        P=H,
        A=A,
        l=lb,
        u=ub,
        QP.settings...
    )

    results = OSQP.solve!(model)

    if results.info.status != :Solved
        @warn "OSQP did not solve the problem"
    end

    Δxs = [
        results.x[slice(t, QP.dims.x, QP.dims.z)]
            for t = 1:QP.dims.T
    ]

    Δus = [
        results.x[slice(t, QP.dims.x + 1, QP.dims.z, QP.dims.z)]
            for t = 1:QP.dims.T
    ]

    ΔZ = Trajectory(Δxs, Δus, Ẑ.times, Ẑ.T, Ẑ.Δt)

    return ΔZ
end


@inline function build_hessian(
    Q::Float64,
    R::Float64,
    dims::NamedTuple
)
    Hₜ = spdiagm([Q * ones(dims.x); R * ones(dims.u)])
    H = kron(sparse(I(dims.T)), Hₜ)
    return H
end

@inline function build_constraint_matrix(
    QP::QuadraticProblem,
    Ẑ::Trajectory,
    ΔY::MeasurementData,
)
    ∇F = build_dynamics_constraint_jacobian(QP, Ẑ)

    ∇G = build_measurement_constraint_jacobian(
        QP,
        Ẑ,
        ΔY
    )

    C = build_constrols_constraint_matrix(QP)

    f_cons = zeros(QP.dims.x * (QP.dims.T - 1))
    g_cons = -vcat(ΔY.ys...)
    u_lb = -foldr(vcat, fill(QP.u_bounds, Ẑ.T - 1)) -
        vcat(Ẑ.actions[1:Ẑ.T - 1]...)
    u_ub = foldr(vcat, fill(QP.u_bounds, Ẑ.T - 1)) -
        vcat(Ẑ.actions[1:Ẑ.T - 1]...)

    lb = vcat(f_cons, g_cons, u_lb, zeros(QP.dims.u))
    ub = vcat(f_cons, g_cons, u_ub, zeros(QP.dims.u))


    A = sparse_vcat(∇F, ∇G, C)

    return A, lb, ub
end

@inline function build_constrols_constraint_matrix(
    QP::QuadraticProblem
)
    C = spzeros(
        QP.dims.u * QP.dims.T,
        QP.dims.z * QP.dims.T
    )

    for t = 1:QP.dims.T
        C[
            slice(
                t,
                QP.dims.u
            ),
            slice(
                t,
                QP.dims.x + 1,
                QP.dims.z,
                QP.dims.z
            )
        ] = sparse(I(QP.dims.u))
    end

    return C
end

@inline function build_measurement_constraint_jacobian(
    QP::QuadraticProblem,
    Ẑ::Trajectory,
    ΔY::MeasurementData
)
    ∇G = spzeros(QP.dims.y * QP.dims.M, QP.dims.z * QP.dims.T)

    for i = 1:QP.dims.M

        τᵢ = ΔY.times[i]

        ∇gᵢ = QP.∇g(Ẑ.states[τᵢ])

        if QP.correction_term
            ∇²gᵢ = QP.∇²g(Ẑ.states[τᵢ])
            ϵ̂ᵢ = pinv(∇gᵢ) * ΔY.ys[i]
            @einsum ∇gᵢ[j, k] += ∇²gᵢ[j, k, l] * ϵ̂ᵢ[l]
        end

        ∇G[
            slice(i, QP.dims.y),
            slice(τᵢ, QP.dims.x, QP.dims.z)
        ] = sparse(∇gᵢ)
    end

    return ∇G
end

# TODO: add feat to just store jacobian of goal traj
@inline function build_dynamics_constraint_jacobian(
    QP::QuadraticProblem,
    Ẑ::Trajectory
)
    ∇f = Jacobian(QP.f)

    ∇F = spzeros(
        QP.dims.x * (QP.dims.T - 1),
        QP.dims.z * QP.dims.T
    )

    for t = 1:QP.dims.T - 1
        ∇fₜ = spzeros(QP.dims.x, QP.dims.z + QP.dims.x)

        xₜ = Ẑ.states[t]
        uₜ = Ẑ.actions[t]
        xₜ₊₁ = Ẑ.states[t + 1]

        ∂xₜfₜ = ∇f(uₜ, Ẑ.Δt, false)
        ∇fₜ[1:QP.dims.x, 1:QP.dims.x] = sparse(∂xₜfₜ)

        for j = 1:QP.dims.u
            ∂uʲₜfₜ = ∇f(xₜ₊₁, xₜ, uₜ, Ẑ.Δt, j)
            ∇fₜ[1:QP.dims.x, QP.dims.x + j] = sparse(∂uʲₜfₜ)
        end

        ∂xₜ₊₁fₜ = ∇f(uₜ, Ẑ.Δt, true)
        ∇fₜ[1:QP.dims.x, QP.dims.z .+ (1:QP.dims.x)] =
            sparse(∂xₜ₊₁fₜ)

        ∇F[
            slice(t, QP.dims.x),
            slice(t, QP.dims.z; stretch=QP.dims.x)
        ] = ∇fₜ
    end

    return ∇F
end




mutable struct ILCProblem
    Ẑgoal::Trajectory
    Ẑ::Trajectory
    QP::QuadraticProblem
    Ygoal::MeasurementData
    Ȳs::Vector{MeasurementData}
    Us::Vector{Matrix{Float64}}
    experiment::QuantumExperiment
    settings::Dict

    function ILCProblem(
        sys::QuantumSystem,
        Ẑgoal::Trajectory,
        experiment::QuantumExperiment;
        integrator=:FourthOrderPade,
        Q=1.0,
        R=1.0,
        u_bounds=[2π * 0.5 for j = 1:sys.ncontrols],
        correction_term=true,
        verbose=true,
        max_iter=100,
        tol=1e-6,
        norm_p=Inf,
        QP_max_iter=1000,
        QP_verbose=false,
        QP_settings=Dict(),
    )
        @assert length(u_bounds) == sys.ncontrols

        ILC_settings = Dict(
            :max_iter => max_iter,
            :tol => tol,
            :norm_p => norm_p,
            :verbose => verbose,
        )

        QP_kw_settings = Dict(
            :max_iter => QP_max_iter,
            :verbose => QP_verbose,
        )

        QP_settings = merge(QP_kw_settings, QP_settings)

        f = eval(integrator)(sys)

        xdim = size(Ẑgoal.states[1], 1)
        udim = size(Ẑgoal.actions[1], 1)

        QP = QuadraticProblem(
            f, experiment.g,
            Q, R,
            u_bounds,
            correction_term,
            QP_settings,
            xdim,
            udim,
            experiment.ydim,
            Ẑgoal.T,
            length(experiment.τs)
        )

        Ygoal = measure(
            Ẑgoal,
            experiment.g,
            experiment.τs,
            experiment.ydim
        )

        return new(
            Ẑgoal,
            Ẑgoal,
            QP,
            Ygoal,
            MeasurementData[],
            Matrix{Float64}[],
            experiment,
            ILC_settings
        )
    end
end

function ProblemSolvers.solve!(prob::ILCProblem)
    U = prob.Ẑ.actions
    Ȳ = prob.experiment(U)
    push!(prob.Ȳs, Ȳ)
    push!(prob.Us, hcat(U...))
    ΔY = Ȳ - prob.Ygoal
    k = 1
    while norm(ΔY, prob.settings[:norm_p]) > prob.settings[:tol]
        if k > prob.settings[:max_iter]
            @info "max iterations reached" max_iter = prob.settings[:max_iter] "|ΔY|" = norm(ΔY, prob.settings[:norm_p]) tol = prob.settings[:tol]
            return
        end
        if prob.settings[:verbose]
            println("iteration = ", k)
            println("|ΔY| = ", norm(ΔY, prob.settings[:norm_p]))
            println()
        end
        ΔZ = prob.QP(prob.Ẑ, ΔY)
        prob.Ẑ = prob.Ẑ + ΔZ
        U = prob.Ẑ.actions
        Ȳ = prob.experiment(U)
        push!(prob.Ȳs, Ȳ)
        push!(prob.Us, hcat(U...))
        ΔY = Ȳ - prob.Ygoal
        k += 1
    end
    @info "ILC converged!" Symbol("|ΔY|") = norm(ΔY, prob.settings[:norm_p]) tol = prob.settings[:tol] iter = k
end



end
