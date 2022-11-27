module IterativeLearningControl

export ILCProblem
export solve!

export QuadraticProblem

export MeasurementData
export measure

export AbstractExperiment
export HardwareExperiment
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
using Statistics
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
    ψ̃::Vector{Vector{Float64}},
    g::Function,
    τs::AbstractVector{Int},
    ydim::Int,
)
    @assert size(g(ψ̃[1]), 1) == ydim
    ys = Vector{Vector{Float64}}(undef, length(τs))
    for (i, τ) in enumerate(τs)
        ys[i] = g(ψ̃[τ])
    end
    return MeasurementData(ys, τs, ydim)
end

abstract type AbstractExperiment end

struct HardwareExperiment <: AbstractExperiment
    g_hardware::Function
    g_analytic::Function
    τs::AbstractVector{Int}
    u_times::Vector{Float64}
    ydim::Int
end

function (experiment::HardwareExperiment)(
    us::Vector{Vector{Float64}};
    backtracking=false
)::MeasurementData
    if !backtracking
        return experiment.g_hardware(us, experiment.u_times, experiment.τs)
    else
        return experiment.g_hardware(us, experiment.u_times, [experiment.τs[end]])
    end
end

struct QuantumExperiment <: AbstractExperiment
    ψ̃₁::Vector{Float64}
    ts::Vector{Float64}
    g::Function
    ydim::Int
    τs::AbstractVector{Int}
    integrator::Function
    G_drift::AbstractMatrix{Float64}
    G_drives::Vector{AbstractMatrix{Float64}}
    G_error_term::AbstractMatrix{Float64}
end

function QuantumExperiment(
    sys::QuantumSystem,
    ψ̃₁::Vector{Float64},
    ts::Vector{Float64},
    g::Function,
    τs::AbstractVector{Int};
    G_error_term=zeros(size(sys.G_drift)),
    integrator=exp
)
    ydim = size(g(ψ̃₁), 1)
    return QuantumExperiment(
        ψ̃₁,
        ts,
        g,
        ydim,
        τs,
        integrator,
        sys.G_drift,
        sys.G_drives,
        G_error_term
    )
end

# TODO:
# - add noise terms (must correspond to ketdim)
# - add multiple quantum state functionality here
# - show fidelity

function (experiment::QuantumExperiment)(
    U::Vector{Vector{Float64}};
    backtracking=false
)::MeasurementData

    T = length(U)

    Ψ̃ = Vector{typeof(experiment.ψ̃₁)}(undef, T)
    Ψ̃[1] = experiment.ψ̃₁

    for t = 2:T
        Gₜ = Integrators.G(
            U[t - 1],
            experiment.G_drift,
            experiment.G_drives
        ) + experiment.G_error_term

        Δt = experiment.ts[t] - experiment.ts[t - 1]

        Ψ̃[t] = experiment.integrator(Gₜ * Δt) * Ψ̃[t - 1]
    end

    if backtracking 
        Ȳ = measure(
            Ψ̃,
            experiment.g,
            experiment.τs[end:end],
            experiment.ydim
        )
    else
        Ȳ = measure(
            Ψ̃,
            experiment.g,
            experiment.τs,
            experiment.ydim
        )
    end


    return Ȳ
end


# TODO:
# - make more general for arbitrary dynamics
#   - make work with augmented state
# - add functionality to store non updating jacobians
# - OSQP error handling

abstract type QuadraticProblem end

struct StaticQuadraticProblem <: QuadraticProblem
    Hreg::SparseMatrixCSC
    A::SparseMatrixCSC
    ∂gs::Vector{Matrix{Float64}}
    ∂²gs::Vector{Array{Float64}}
    Σinv::Symmetric{Float64}
    Qy::Float64
    Qf::Float64
    u_bounds::Vector{Float64}
    correction_term::Bool
    mle::Bool
    dims::NamedTuple
    settings::Dict{Symbol, Any}
end

function StaticQuadraticProblem(
    Ẑgoal::Trajectory,
    f::Function,
    g::Function,
    Q::Float64,
    Qy::Float64,
    Qf::Float64,
    R::Float64,
    Σ::AbstractMatrix{Float64},
    u_bounds::Vector{Float64},
    correction_term::Bool,
    settings::Dict{Symbol, Any},
    dims::NamedTuple;
    mle=true
)
    @assert size(Σ, 1) == size(Σ, 2) == dims.y

    ∂f(zz) = ForwardDiff.jacobian(f, zz)

    ∂g(x) = ForwardDiff.jacobian(g, x)

    function ∂²g(x)
        H = ForwardDiff.jacobian(u -> vec(∂g(u)), x)
        return reshape(H, dims.y, dims.x, dims.x)
    end

    ∂F = build_dynamics_constraint_jacobian(
        Ẑgoal,
        ∂f,
        dims
    )

    ∂gs = Matrix[]

    for t = 1:Ẑgoal.T
        push!(∂gs, ∂g(Ẑgoal.states[t]))
    end

    ∂²gs = Array[]

    for t = 1:Ẑgoal.T
        push!(∂²gs, ∂²g(Ẑgoal.states[t]))
    end

    C_u = build_controls_constraint_matrix(dims)
    C_x₁ = build_initial_state_constraint_matrix(dims)

    A = sparse_vcat(∂F, C_u, C_x₁)

    Hreg = build_regularization_hessian(Q, R, dims)

    Σinv = Symmetric(inv(Σ))

    return StaticQuadraticProblem(
        Hreg,
        A,
        ∂gs,
        ∂²gs,
        Σinv,
        Qy,
        Qf,
        u_bounds,
        correction_term,
        mle,
        dims,
        settings
    )
end

function (QP::StaticQuadraticProblem)(
    Ẑ::Trajectory,
    ΔY::MeasurementData
)
    model = OSQP.Model()

    C_u_lb, C_u_ub = build_controls_constraint_bounds(Ẑ, QP.u_bounds, QP.dims)

    ∂F_cons = zeros(QP.dims.x * (QP.dims.T - 1))

    C_x₁_cons = zeros(QP.dims.x)

    lb = vcat(∂F_cons, C_u_lb, C_x₁_cons)
    ub = vcat(∂F_cons, C_u_ub, C_x₁_cons)

    if QP.mle
        Hmle, ∇ = build_mle_hessian_and_gradient(
            ΔY, QP.∂gs, QP.∂²gs, QP.Qy, QP.Qf, QP.Σinv, QP.dims, QP.correction_term
        )
        H = Hmle + QP.Hreg
        A = QP.A
    else
        H = Hreg
        ∂G, ∂G_cons = build_measurement_constraint_jacobian(
            ΔY, QP.∂gs, QP.∂²gs, QP.dims, QP.correction_term
        )
        A = sparse_vcat(∂G, QP.A)
        ub = vcat(∂G_cons, ub)
        lb = vcat(∂G_cons, lb)
    end

    OSQP.setup!(
        model;
        P=H,
        A=A,
        q=QP.mle ? ∇ : nothing,
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



struct DynamicQuadraticProblem <: QuadraticProblem
    ∂f::Function
    ∂g::Function
    ∂²g::Function
    Q::Float64
    Qy::Float64
    Qf::Float64
    R::Float64
    Σinv::Union{Symmetric{Float64}, Nothing}
    u_bounds::Vector{Float64}
    correction_term::Bool
    settings::Dict{Symbol, Any}
    dims::NamedTuple
    mle::Bool
end

function DynamicQuadraticProblem(
    f::Function,
    g::Function,
    Q::Float64,
    Qy::Float64,
    Qf::Float64,
    R::Float64,
    Σ::AbstractMatrix{Float64},
    u_bounds::Vector{Float64},
    correction_term::Bool,
    settings::Dict{Symbol, Any},
    dims::NamedTuple
)
    @assert size(Σ, 1) == size(Σ, 2) == dims.y

    ∂f(zz) = ForwardDiff.jacobian(f, zz)

    ∂g(x) = ForwardDiff.jacobian(g, x)

    function ∂²g(x)
        H = ForwardDiff.jacobian(u -> vec(∂g(u)), x)
        return reshape(H, dims.y, dims.x, dims.x)
    end

    Σinv = Symmetric(inv(Σ))

    return DynamicQuadraticProblem(
        ∂f,
        ∂g,
        ∂²g,
        Q,
        Qy,
        Qf,
        R,
        Σinv,
        u_bounds,
        correction_term,
        settings,
        dims,
        !isnothing(Σ)
    )
end

function (QP::DynamicQuadraticProblem)(
    Ẑ::Trajectory,
    ΔY::MeasurementData
)
    model = OSQP.Model()

    Hreg = build_regularization_hessian(QP.Q, QP.R, QP.dims)

    ∂F = build_dynamics_constraint_jacobian(Ẑ, QP.∂f, QP.dims)
    ∂F_cons = zeros(size(∂F, 1))

    C_u = build_controls_constraint_matrix(QP.dims)
    C_u_lb, C_u_ub = build_controls_constraint_bounds(Ẑ, QP.u_bounds, QP.dims)

    C_x₁ = build_initial_state_constraint_matrix(QP.dims)
    C_x₁_cons = zeros(size(C_x₁, 1))

    A = sparse_vcat(∂F, C_u, C_x₁)

    lb = vcat(∂F_cons, C_u_lb, C_x₁_cons)
    ub = vcat(∂F_cons, C_u_ub, C_x₁_cons)

    if QP.mle
        Hmle, ∇ = build_mle_hessian_and_gradient(
            Ẑ, ΔY, QP.∂g, QP.∂²g, QP.Qy, QP.Qf, QP.Σinv, QP.dims, QP.correction_term
        )
        H = Hmle + Hreg
    else
        ∂G, ∂G_cons = build_measurement_constraint_jacobian(
            Ẑ, ΔY, QP.∂g, QP.∂²g, QP.dims, QP.correction_term
        )
        A = sparse_vcat(∂G, A)
        lb = vcat(∂G_cons, lb)
        ub = vcat(∂G_cons, ub)
        H = Hreg
    end

    OSQP.setup!(
        model;
        P=H,
        A=A,
        q=QP.mle ? ∇ : nothing,
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


@inline function build_regularization_hessian(
    Q::Float64,
    R::Float64,
    dims::NamedTuple
)
    Hₜ = spdiagm([Q * ones(dims.x); R * ones(dims.u)])
    H = kron(sparse(I(dims.T)), sparse(Hₜ))
    return H
end

@inline function build_mle_hessian_and_gradient(
    Ẑ::Trajectory,
    ΔY::MeasurementData,
    ∂g::Function,
    ∂²g::Function,
    Q::Float64,
    Qf::Float64,
    Σinv::AbstractMatrix,
    dims::NamedTuple,
    correction_term::Bool
)
    Hmle = spzeros(dims.z * dims.T, dims.z * dims.T)

    ∇ = zeros(dims.z * dims.T)

    for i = 1:dims.M

        τᵢ = ΔY.times[i]

        ∂gᵢ = ∂g(Ẑ.states[τᵢ])

        if correction_term
            ∂²gᵢ = ∂²g(Ẑ.states[τᵢ])
            ϵ̂ᵢ = pinv(∂gᵢ) * ΔY.ys[i]
            @einsum ∂gᵢ[j, k] += ∂²gᵢ[j, k, l] * ϵ̂ᵢ[l]
        end

        Hᵢmle = ∂gᵢ' * Σinv * ∂gᵢ

        Hmle[
            slice(τᵢ, dims.x, dims.z),
            slice(τᵢ, dims.x, dims.z)
        ] = sparse(Hᵢmle)

        ∇ᵢmle = ΔY.ys[i]' * Σinv * ∂gᵢ

        if i == dims.M
            Hᵢmle *= Qf
            ∇ᵢmle *= Qf
        else
            Hᵢmle *= Q
            ∇ᵢmle *= Q
        end

        ∇[slice(τᵢ, dims.x, dims.z)] = ∇ᵢmle
    end

    return Hmle, ∇
end

@inline function build_mle_hessian_and_gradient(
    ΔY::MeasurementData,
    ∂gs::Vector{Matrix{Float64}},
    ∂²gs::Vector{Array{Float64}},
    Qy::Float64,
    Qf::Float64,
    Σinv::AbstractMatrix,
    dims::NamedTuple,
    correction_term::Bool
)
    Hmle = spzeros(dims.z * dims.T, dims.z * dims.T)

    ∇ = zeros(dims.z * dims.T)

    for i = 1:dims.M

        τᵢ = ΔY.times[i]

        ∂gᵢ = ∂gs[i]

        if correction_term
            ∂²gᵢ = ∂²gs[i]
            ϵ̂ᵢ = pinv(∂gᵢ) * ΔY.ys[i]
            @einsum ∂gᵢ[j, k] += ∂²gᵢ[j, k, l] * ϵ̂ᵢ[l]
        end

        Hᵢmle = ∂gᵢ' * Σinv * ∂gᵢ

        Hmle[
            slice(τᵢ, dims.x, dims.z),
            slice(τᵢ, dims.x, dims.z)
        ] = sparse(Hᵢmle)

        ∇ᵢmle = ΔY.ys[i]' * Σinv * ∂gᵢ

        if i == dims.M
            Hᵢmle *= Qf
            ∇ᵢmle *= Qf
        else
            Hᵢmle *= Qy
            ∇ᵢmle *= Qy
        end

        ∇[slice(τᵢ, dims.x, dims.z)] = ∇ᵢmle
    end

    return Hmle, ∇
end

@inline function build_initial_state_constraint_matrix(
    dims::NamedTuple
)
    C_x₁ = sparse_hcat(
        I(dims.x),
        spzeros(dims.x, dims.u + dims.z * (dims.T - 1))
    )
    return C_x₁
end


@inline function build_controls_constraint_matrix(
    dims::NamedTuple
)
    C_u = spzeros(
        dims.u * dims.T,
        dims.z * dims.T
    )

    for t = 1:dims.T
        C_u[
            slice(
                t,
                dims.u
            ),
            slice(
                t,
                dims.x + 1,
                dims.z,
                dims.z
            )
        ] = sparse(I(dims.u))
    end
    return C_u
end

@inline function build_controls_constraint_bounds(
    Ẑ::Trajectory,
    u_bounds::Vector,
    dims::NamedTuple
)
    C_u_lb = -foldr(vcat, fill(u_bounds, dims.T - 2)) -
        vcat(Ẑ.actions[2:dims.T - 1]...)

    C_u_ub = foldr(vcat, fill(u_bounds, dims.T - 2)) -
        vcat(Ẑ.actions[2:dims.T - 1]...)

    C_u_lb = [zeros(dims.u); C_u_lb; zeros(dims.u)]

    C_u_ub = [zeros(dims.u); C_u_ub; zeros(dims.u)]

    return C_u_lb, C_u_ub
end

@inline function build_measurement_constraint_jacobian(
    Ẑ::Trajectory,
    ΔY::MeasurementData,
    ∂g::Function,
    ∂²g::Function,
    dims::NamedTuple,
    correction_term::Bool
)
    ∂G = spzeros(dims.y * dims.M, dims.z * dims.T)

    for i = 1:dims.M

        τᵢ = ΔY.times[i]

        ∂gᵢ = ∂g(Ẑ.states[τᵢ])

        if correction_term
            ∂²gᵢ = ∂²g(Ẑ.states[τᵢ])
            ϵ̂ᵢ = pinv(∂gᵢ) * ΔY.ys[i]
            @einsum ∂gᵢ[j, k] += ∂²gᵢ[j, k, l] * ϵ̂ᵢ[l]
        end

        ∂G[
            slice(i, dims.y),
            slice(τᵢ, dims.x, dims.z)
        ] = sparse(∂gᵢ)
    end

    ∂G_cons = -vcat(ΔY.ys...)

    return ∂G, ∂G_cons
end

@inline function build_measurement_constraint_jacobian(
    ΔY::MeasurementData,
    ∂gs::Vector{Matrix},
    ∂²gs::Vector{Array},
    dims::NamedTuple,
    correction_term::Bool
)
    ∂G = spzeros(dims.y * dims.M, dims.z * dims.T)

    for i = 1:dims.M

        τᵢ = ΔY.times[i]

        ∂gᵢ = ∂gs[i]

        if correction_term
            ∂²gᵢ = ∂²gs[i]
            ϵ̂ᵢ = pinv(∂gᵢ) * ΔY.ys[i]
            @einsum ∂gᵢ[j, k] += ∂²gᵢ[j, k, l] * ϵ̂ᵢ[l]
        end

        ∂G[
            slice(i, dims.y),
            slice(τᵢ, dims.x, dims.z)
        ] = sparse(∂gᵢ)
    end

    ∂G_cons = -vcat(ΔY.ys...)

    return ∂G, ∂G_cons
end
# TODO: add feat to just store jacobian of goal traj
@inline function build_dynamics_constraint_jacobian(
    Ẑ::Trajectory,
    ∂f::Function,
    dims::NamedTuple
)
    ∂F = spzeros(
        dims.x * (dims.T - 1),
        dims.z * dims.T
    )

    for t = 1:dims.T - 1

        zₜzₜ₊₁ = [
            Ẑ.states[t];
            Ẑ.actions[t];
            Ẑ.states[t + 1];
            Ẑ.actions[t + 1]
        ]

        ∂F[
            slice(t, dims.x),
            slice(t, dims.z; stretch=dims.z)
        ] = sparse(∂f(zₜzₜ₊₁))
    end

    return ∂F
end


function random_cov_matrix(n::Int; σ=1.0)
    σs = σ * rand(n)
    sort!(σs, rev=true)
    Q = qr(randn(n, n)).Q
    return Q * Diagonal(σs) * Q'
end




mutable struct ILCProblem
    Ẑgoal::Trajectory
    Ẑ::Trajectory
    QP::QuadraticProblem
    Ygoal::MeasurementData
    Ȳs::Vector{MeasurementData}
    Us::Vector{Matrix{Float64}}
    experiment::AbstractExperiment
    settings::Dict{Symbol, Any}
    bt_dict::Dict{Int, Vector}

    function ILCProblem(
        sys::QuantumSystem,
        Ẑgoal::Trajectory,
        experiment::HardwareExperiment;
        integrator=:FourthOrderPade,
        Q=0.0,
        Qy=1.0,
        Qf=100.0,
        R=1.0,
        # identity matrix of Float64
        Σ=Diagonal(fill(1.0, experiment.ydim)),
        u_bounds=sys.a_bounds,
        correction_term=true,
        verbose=true,
        max_iter=100,
        max_backtrack_iter=10,
        tol=1e-6,
        α=0.5,
        β=0.5,
        norm_p=Inf,
        static_QP=false,
        QP_max_iter=100_000,
        QP_verbose=false,
        QP_settings=Dict(),
        use_system_goal=false,
    )
        @assert length(u_bounds) == sys.ncontrols

        ILC_settings::Dict{Symbol, Any} = Dict(
            :max_iter => max_iter,
            :max_backtrack_iter => max_backtrack_iter,
            :α => α,
            :β => β,
            :tol => tol,
            :norm_p => norm_p,
            :verbose => verbose,
        )

        QP_kw_settings::Dict{Symbol, Any} = Dict(
            :max_iter => QP_max_iter,
            :verbose => QP_verbose,
        )

        QP_settings::Dict{Symbol, Any} =
            merge(QP_kw_settings, QP_settings)

        dynamics = eval(integrator)(sys)

        dims = (
            x=size(Ẑgoal.states[1], 1),
            u=size(Ẑgoal.actions[1], 1),
            z=size(Ẑgoal.states[1], 1) + size(Ẑgoal.actions[1], 1),
            y=experiment.ydim,
            T=Ẑgoal.T,
            M=length(experiment.τs)
        )

        f = zₜzₜ₊₁ -> begin
            xₜ = zₜzₜ₊₁[1:dims.x]
            uₜ = zₜzₜ₊₁[dims.x .+ (1:dims.u)]
            xₜ₊₁ = zₜzₜ₊₁[dims.z .+ (1:dims.x)]
            return dynamics(xₜ₊₁, xₜ, uₜ, Ẑgoal.Δt)
        end

        if static_QP
            QP = StaticQuadraticProblem(
                Ẑgoal,
                f, experiment.g_analytic,
                Q, Qy, Qf, R, Σ,
                u_bounds,
                correction_term,
                QP_settings,
                dims
            )
        else
            QP = DynamicQuadraticProblem(
                f, experiment.g_analytic,
                Q, Qy, Qf, R, Σ,
                u_bounds,
                correction_term,
                QP_settings,
                dims
            )
        end

        Ygoal = measure(
            Ẑgoal,
            experiment.g_analytic,
            experiment.τs,
            experiment.ydim
        )

        if use_system_goal
            Ygoal.ys[end] = experiment.g_analytic(sys.ψ̃goal[slice(1, sys.isodim)])
        end

        # display(Ygoal.ys[end])
        # println()
        # display(Ẑgoal.states[end])

        return new(
            Ẑgoal,
            Ẑgoal,
            QP,
            Ygoal,
            MeasurementData[],
            Matrix{Float64}[],
            experiment,
            ILC_settings,
            Dict{Int, Vector}()
        )
    end

    function ILCProblem(
        sys::QuantumSystem,
        Ẑgoal::Trajectory,
        experiment::QuantumExperiment;
        integrator=:FourthOrderPade,
        Q=0.0,
        Qy=1.0,
        Qf=100.0,
        R=1.0,
        # identity matrix of Float64
        Σ=Diagonal(fill(1.0, experiment.ydim)),
        u_bounds=sys.a_bounds,
        correction_term=true,
        verbose=true,
        max_iter=100,
        max_backtrack_iter=10,
        tol=1e-6,
        α=0.5,
        β=0.5,
        norm_p=Inf,
        static_QP=false,
        QP_max_iter=100_000,
        QP_verbose=false,
        QP_settings=Dict(),
        use_system_goal=false,
    )
        @assert length(u_bounds) == sys.ncontrols

        ILC_settings::Dict{Symbol, Any} = Dict(
            :max_iter => max_iter,
            :max_backtrack_iter => max_backtrack_iter,
            :α => α,
            :β => β,
            :tol => tol,
            :norm_p => norm_p,
            :verbose => verbose,
        )

        QP_kw_settings::Dict{Symbol, Any} = Dict(
            :max_iter => QP_max_iter,
            :verbose => QP_verbose,
        )

        QP_settings::Dict{Symbol, Any} =
            merge(QP_kw_settings, QP_settings)

        dynamics = eval(integrator)(sys)

        dims = (
            x=size(Ẑgoal.states[1], 1),
            u=size(Ẑgoal.actions[1], 1),
            z=size(Ẑgoal.states[1], 1) + size(Ẑgoal.actions[1], 1),
            y=experiment.ydim,
            T=Ẑgoal.T,
            M=length(experiment.τs)
        )

        f = zₜzₜ₊₁ -> begin
            xₜ = zₜzₜ₊₁[1:dims.x]
            uₜ = zₜzₜ₊₁[dims.x .+ (1:dims.u)]
            xₜ₊₁ = zₜzₜ₊₁[dims.z .+ (1:dims.x)]
            return dynamics(xₜ₊₁, xₜ, uₜ, Ẑgoal.Δt)
        end

        if static_QP
            QP = StaticQuadraticProblem(
                Ẑgoal,
                f, experiment.g,
                Q, Qy, Qf, R, Σ,
                u_bounds,
                correction_term,
                QP_settings,
                dims
            )
        else
            QP = DynamicQuadraticProblem(
                f, experiment.g,
                Q, Qy, Qf, R, Σ,
                u_bounds,
                correction_term,
                QP_settings,
                dims
            )
        end

        Ygoal = measure(
            Ẑgoal,
            experiment.g,
            experiment.τs,
            experiment.ydim
        )

        if use_system_goal
            Ygoal.ys[end] = experiment.g(sys.ψ̃goal[slice(1, sys.isodim)])
        end

        # display(Ygoal.ys[end])
        # println()
        # display(Ẑgoal.states[end])

        return new(
            Ẑgoal,
            Ẑgoal,
            QP,
            Ygoal,
            MeasurementData[],
            Matrix{Float64}[],
            experiment,
            ILC_settings,
            Dict{Int, Vector}()
        )
    end
end

function fidelity(ψ̃, ψ̃goal)
    ψ = iso_to_ket(ψ̃)
    ψgoal = iso_to_ket(ψ̃goal)
    # println("norm(ψ)     = $(norm(ψ))")
    # println("norm(ψgoal) = $(norm(ψgoal))")
    return abs2(ψ' * ψgoal)
end


function ProblemSolvers.solve!(prob::ILCProblem)
    U = prob.Ẑ.actions
    Ȳ = prob.experiment(U)
    push!(prob.Ȳs, Ȳ)
    push!(prob.Us, hcat(U...))
    ΔY = Ȳ - prob.Ygoal
    println()
    printstyled()
    ΔyT_norms = [norm(ΔY.ys[end], prob.settings[:norm_p])]
    k = 1
    while true 
        if k > prob.settings[:max_iter]
            @info "max iterations reached" max_iter = prob.settings[:max_iter] 
            return
        end
        if prob.settings[:verbose]
            println()
            printstyled("iter    = ", k; color=:green)
            println()
            printstyled("⟨|ΔY|⟩  = ", mean([norm(y, prob.settings[:norm_p]) for y in ΔY.ys]); color=:green)
            println()
            printstyled("|ΔY(T)| = ", norm(ΔY.ys[end], prob.settings[:norm_p]); color=:green)
            println()
            println()
        end
        ΔZ = prob.settings[:β] * prob.QP(prob.Ẑ, ΔY)
        prob.Ẑ = prob.Ẑ + ΔZ

        U = prob.Ẑ.actions
        Ȳ = prob.experiment(U)
        ΔYnext = Ȳ - prob.Ygoal
        ΔyTnext = ΔYnext.ys[end]

        # backtracking
        if norm(ΔyTnext, prob.settings[:norm_p]) >
            minimum(ΔyT_norms)

            printstyled("   backtracking"; color=:magenta)
            println()
            println()
            i = 1
            backtrack_yts = []

            iter_ΔyT_norms = []

            while norm(ΔyTnext, prob.settings[:norm_p]) >
                minimum(ΔyT_norms)
                # norm(ΔY.ys[end], prob.settings[:norm_p])
                if i > prob.settings[:max_backtrack_iter]
                    println()
                    printstyled("   max backtracking iterations reached"; color=:magenta)
                    println()
                    println()
                    ΔY = ΔYnext
                    return
                end
                prob.Ẑ = prob.Ẑ - ΔZ
                ΔZ = prob.settings[:α] * ΔZ
                prob.Ẑ = prob.Ẑ + ΔZ
                U = prob.Ẑ.actions

                yTnext = prob.experiment(U; backtracking=true).ys[end]
                ΔyTnext = yTnext - prob.Ygoal.ys[end]

                push!(backtrack_yts, yTnext)
                push!(iter_ΔyT_norms, norm(ΔyTnext, prob.settings[:norm_p]))

                println()
                printstyled("       bt_iter     = ", i; color=:cyan)
                println()
                printstyled("       min |ΔY(T)| = ", minimum(ΔyT_norms); color=:cyan)
                println()
                printstyled("       |ΔY(T)|     = ", norm(ΔyTnext, prob.settings[:norm_p]); color=:cyan)
                println()
                println()

                i += 1
            end
            push!(ΔyT_norms, minimum(iter_ΔyT_norms))
            prob.bt_dict[k] = backtrack_yts

            U = prob.Ẑ.actions
            # remeasure with new controls to get full measurement
            Ȳ_bt = prob.experiment(U)
            ΔY = Ȳ_bt - prob.Ygoal 

            # push remeasured norm(ΔyT) to tracked errors
            push!(ΔyT_norms, norm(ΔY.ys[end], prob.settings[:norm_p]))
        else
            ΔY = ΔYnext
            push!(ΔyT_norms, norm(ΔY.ys[end], prob.settings[:norm_p]))
        end
        push!(prob.Ȳs, Ȳ)
        push!(prob.Us, hcat(U...))
        k += 1
    end
    @info "ILC converged!" "|ΔY|" = norm(ΔY, prob.settings[:norm_p]) tol = prob.settings[:tol] iter = k
end



end
