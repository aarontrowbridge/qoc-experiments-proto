module ILCQuadraticProblems

export QuadraticProblem

export StaticQuadraticProblem
export DynamicQuadraticProblem

export StaticQuadraticProblemNew
export DynamicQuadraticProblemNew

using ..Trajectories
using ..ILCExperiments
using ..ILCTrajectories
using ..Utils

using LinearAlgebra
using SparseArrays
using ForwardDiff
using Einsum
using OSQP

abstract type QuadraticProblem end

struct StaticQuadraticProblemNew <: QuadraticProblem
    Hreg::SparseMatrixCSC
    A::SparseMatrixCSC
    ∂gs::Vector{Matrix{Float64}}
    ∂²gs::Vector{Array{Float64}}
    ∂g::Function
    ∂²g::Function
    Qy::Float64
    Qyf::Float64
    R::NamedTuple{names, <:Tuple{Vararg{Float64}}} where names
    correction_term::Bool
    dynamic_measurements::Bool
    mle::Bool
    settings::Dict{Symbol, Any}
end

function StaticQuadraticProblemNew(
    Ẑgoal::Traj,
    f::Function,
    g::Function,
    τs::AbstractVector{Int},
    Qy::Float64,
    Qyf::Float64,
    R::NamedTuple{names, <:Tuple{Vararg{Float64}}} where names,
    correction_term::Bool,
    dynamic_measurements::Bool,
    settings::Dict{Symbol, Any};
    mle::Bool=true
)
    @assert :ψ̃ ∈ keys(Ẑgoal.components) "must specify quantum states component as ψ̃"
    @assert all([k ∈ keys(Ẑgoal.components) for k in keys(R)]) "regularization multiplier keys must match trajectory component keys"

    ydim = length(g(Ẑgoal[1].ψ̃))


    # dynamics constraints

    @assert length(f(vec(Ẑgoal.data[:, 1:2]))) == Ẑgoal.dims.states

    ∂f(zz) = ForwardDiff.jacobian(f, zz)

    ∂F = build_dynamics_constraint_jacobian(Ẑgoal, ∂f)


    # measurement constraints

    ∂g(x) = ForwardDiff.jacobian(g, x)

    function ∂²g(x)
        H = ForwardDiff.jacobian(u -> vec(∂g(u)), x)
        return reshape(H, ydim, Ẑgoal.dims.ψ̃, Ẑgoal.dims.ψ̃)
    end

    ∂gs = Matrix[]

    for τ ∈ τs
        push!(∂gs, ∂g(Ẑgoal[τ].ψ̃))
    end

    ∂²gs = Array[]

    for τ ∈ τs
        push!(∂²gs, ∂²g(Ẑgoal[τ].ψ̃))
    end


    # bounds constraints
    B = build_bounds_constraint_matrix(Ẑgoal)

    # initial/final value constraints
    C = build_initial_and_final_constraint_matrix(Ẑgoal)

    # full constraint matrix
    A = sparse_vcat(∂F, B, C)

    # hessian of regularization objective terms
    Hreg = build_regularization_hessian(R, Ẑgoal)

    return StaticQuadraticProblemNew(
        Hreg,
        A,
        ∂gs,
        ∂²gs,
        ∂g,
        ∂²g,
        Qy,
        Qyf,
        R,
        correction_term,
        dynamic_measurements,
        mle,
        settings
    )
end

function (QP::StaticQuadraticProblemNew)(
    ΔY::MeasurementData,
    Ẑ::Traj
)

    # constraint values
    lb, ub = build_constraints(Ẑ)

    if QP.mle
        Σinvs = [inv(Σ) for Σ ∈ ΔY.Σs]
        if QP.dynamic_measurements
            Hmle, ∇ = build_mle_hessian_and_gradient_new(
                ΔY,
                QP.∂g,
                QP.∂²g,
                QP.Qy,
                QP.Qyf,
                Σinvs,
                Ẑ.dims,
                Ẑ.components.ψ̃,
                Ẑ.T,
                QP.correction_term
            )
        else
            Hmle, ∇ = build_mle_hessian_and_gradient_new(
                ΔY,
                QP.∂gs,
                QP.∂²gs,
                QP.Qy,
                QP.Qyf,
                Σinvs,
                Ẑ.dims,
                Ẑ.components.ψ̃,
                Ẑ.T,
                QP.correction_term
            )
        end

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

    model = OSQP.Model()

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
        @warn "OSQP did not solve the problem. status: $(results.info.status)"
        println()
    end

    ΔZ = Traj(results.x, Ẑ)

    return ΔZ
end


struct DynamicQuadraticProblemNew <: QuadraticProblem
    ∂f::Function
    ∂g::Function
    ∂²g::Function
    Qy::Float64
    Qyf::Float64
    R::NamedTuple{names, <:Tuple{Vararg{Float64}}} where names
    C::SparseMatrixCSC{Float64, Int64}
    Hreg::SparseMatrixCSC{Float64, Int64}
    correction_term::Bool
    mle::Bool
    settings::Dict{Symbol, Any}
end

function DynamicQuadraticProblemNew(
    Z::Traj,
    f::Function,
    g::Function,
    Qy::Float64,
    Qf::Float64,
    R::NamedTuple{names, <:Tuple{Vararg{Float64}}} where names,
    correction_term::Bool,
    settings::Dict{Symbol, Any};
    mle::Bool=true
)
    @assert :ψ̃ ∈ keys(Z.components) "must specify quantum states component as ψ̃"

    @assert all([k ∈ keys(Z.components) for k in keys(R)])
        "regularization multiplier keys must match trajectory component keys"

    @assert length(f(vec(Z.data[:, 1:2]))) == Z.dims.states

    ∂f(zz) = ForwardDiff.jacobian(f, zz)

    ∂g(x) = ForwardDiff.jacobian(g, x)

    ydim = length(g(Z[1].ψ̃))

    function ∂²g(x)
        H = ForwardDiff.jacobian(u -> vec(∂g(u)), x)
        return reshape(H, ydim, Z.dims.ψ̃, Z.dims.ψ̃)
    end

    C_bounds = build_bounds_constraint_matrix(Z)

    C_init_final = build_initial_and_final_constraint_matrix(Z)

    C = sparse_vcat(C_bounds, C_init_final)

    Hreg = build_regularization_hessian(R, Z)

    return DynamicQuadraticProblemNew(
        ∂f,
        ∂g,
        ∂²g,
        Qy,
        Qf,
        R,
        C,
        Hreg,
        correction_term,
        mle,
        settings
    )
end

function (QP::DynamicQuadraticProblemNew)(
    ΔY::MeasurementData,
    Ẑ::Traj
)
    model = OSQP.Model()

    # get the regularization hessian term
    Hreg = build_regularization_hessian(QP.R, Ẑ)

    # get the dynamics constraint jacobian
    ∂F = build_dynamics_constraint_jacobian(Ẑ, QP.∂f)

    # full constraint matrix
    A = sparse_vcat(∂F, QP.C)

    # get the constraint bounds
    lb, ub = build_constraints(Ẑ)

    if QP.mle

        # get the inverses of the measurment covariances
        Σinvs = [inv(Σ) for Σ ∈ ΔY.Σs]

        # get the hessian and gradient of the mle objective term
        Hmle, ∇ = build_mle_hessian_and_gradient_new(
            ΔY,
            QP.∂g,
            QP.∂²g,
            QP.Qy,
            QP.Qyf,
            Σinvs,
            Ẑ.dims,
            Ẑ.components.ψ̃,
            Ẑ.T,
            QP.correction_term
        )

        # build the full hessian
        H = Hmle + QP.Hreg
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
        @warn "OSQP did not solve the problem, status: $(results.info.status)"
    end

    println("OSQP status: $(results.info.status)")
    println("norm(x): $(norm(results.x))")

    ΔZ = Traj(results.x, Ẑ)

    return ΔZ
end

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
    d2u::Bool
    d2u_bounds::Vector{Float64}
end

function StaticQuadraticProblem(
    Ẑgoal::Trajectory,
    f::Function,
    g::Function,
    τs::AbstractVector{Int},
    Q::Float64,
    Qy::Float64,
    Qf::Float64,
    R::Float64,
    Σ::AbstractMatrix{Float64},
    u_bounds::Vector{Float64},
    correction_term::Bool,
    settings::Dict{Symbol, Any},
    dims::NamedTuple;
    mle=true,
    d2u=false,
    d2u_bounds=fill(1e-5, length(u_bounds))
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

    for τ ∈ τs
        push!(∂gs, ∂g(Ẑgoal.states[τ]))
    end

    ∂²gs = Array[]

    for τ ∈ τs
        push!(∂²gs, ∂²g(Ẑgoal.states[τ]))
    end

    C_u = build_controls_constraint_matrix(dims, d2u=d2u)
    C_x₁ = build_initial_state_constraint_matrix(dims)

    if d2u
        C_d2u = build_d2u_constraint_matrix(dims)
        A = sparse_vcat(∂F, C_u, C_d2u, C_x₁)
    else
        A = sparse_vcat(∂F, C_u, C_x₁)
    end

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
        settings,
        d2u,
        d2u_bounds
    )
end

function (QP::StaticQuadraticProblem)(
    Ẑ::Trajectory,
    ΔY::MeasurementData
)
    model = OSQP.Model()

    C_u_lb, C_u_ub = build_controls_constraint_bounds(Ẑ, QP.u_bounds, QP.dims, d2u=QP.d2u)

    ∂F_cons = zeros(QP.dims.x * (QP.dims.T - 1))

    C_x₁_cons = zeros(QP.dims.x)
    if QP.d2u
        C_d2u_lb, C_d2u_ub = build_d2u_constraint_bounds(Ẑ, QP.dims, d2u_bounds = QP.d2u_bounds)
        lb = vcat(∂F_cons, C_u_lb, C_d2u_lb, C_x₁_cons)
        ub = vcat(∂F_cons, C_u_ub, C_d2u_ub, C_x₁_cons)
    else
        lb = vcat(∂F_cons, C_u_lb, C_x₁_cons)
        ub = vcat(∂F_cons, C_u_ub, C_x₁_cons)
    end

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

    println()
    println("Solving QP...")
    println()
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
    d2u::Bool
    d2u_bounds::Vector{Float64}
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
    dims::NamedTuple;
    d2u=false,
    d2u_bounds=fill(1e-5, length(u_bounds))
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
        !isnothing(Σ),
        d2u,
        d2u_bounds
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

    C_u = build_controls_constraint_matrix(QP.dims, d2u = QP.d2u)
    C_u_lb, C_u_ub = build_controls_constraint_bounds(Ẑ, QP.u_bounds, QP.dims, d2u = QP.d2u)


    C_x₁ = build_initial_state_constraint_matrix(QP.dims)
    C_x₁_cons = zeros(size(C_x₁, 1))

    if QP.d2u
        C_d2u = build_d2u_constraint_matrix(QP.dims)
        C_d2u_lb, C_d2u_ub = build_d2u_constraint_bounds(Ẑ, QP.dims, d2u_bounds = QP.d2u_bounds)

        A = sparse_vcat(∂F, C_u, C_d2u, C_x₁)

        lb = vcat(∂F_cons, C_u_lb, C_d2u_lb, C_x₁_cons)
        ub = vcat(∂F_cons, C_u_ub, C_d2u_ub, C_x₁_cons)
    else
        A = sparse_vcat(∂F, C_u, C_x₁)

        lb = vcat(∂F_cons, C_u_lb, C_x₁_cons)
        ub = vcat(∂F_cons, C_u_ub, C_x₁_cons)
    end

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

    println("norm(x) = $(norm(results.x))")

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

"""
    build_regularization_hessian(
        R::NamedTuple{names, Tuple{Vararg{Float64}}} where names,
        Z::Traj
    )

Builds the hessian matrix for the regularization objective terms.
"""
@inline function build_regularization_hessian(
    R::NamedTuple{names, <:Tuple{Vararg{Float64}}} where names,
    Z::Traj
)
    Hₜ = blockdiag([
        k ∈ keys(R) ?
            R[k] * sparse(I(Z.dims[k])) :
            spzeros(Z.dims[k], Z.dims[k])
        for k ∈ keys(Z.components) if k != :states && k != :controls
    ]...)
    H = kron(sparse(I(Z.T)), Hₜ)
    return H
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

        τᵢ = ΔY.τs[i]

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

        τᵢ = ΔY.τs[i]

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

"""
    build_mle_hessian_and_gradient_new()

Builds the hessian matrix and gradient vector for the MLE objective term.
"""
@inline function build_mle_hessian_and_gradient_new(
    ΔY::MeasurementData,
    ∂gs::Vector{Matrix{Float64}},
    ∂²gs::Vector{Array{Float64}},
    Qy::Float64,
    Qyf::Float64,
    Σinvs::Vector{<:AbstractMatrix},
    dims::NamedTuple,
    measured_components::AbstractVector{Int},
    T::Int, # number of time steps
    correction_term::Bool
)
    M = length(ΔY.τs)
    zdim = dims.states + dims.controls

    # display(Σinvs[1])

    Hmle = spzeros(zdim * T, zdim * T)

    ∇ = zeros(zdim * T)

    for i = 1:M

        τᵢ = ΔY.τs[i]

        ∂gᵢ = ∂gs[i]

        if correction_term
            ∂²gᵢ = ∂²gs[i]
            ϵ̂ᵢ = pinv(∂gᵢ) * ΔY.ys[i]
            @einsum ∂gᵢ[j, k] += ∂²gᵢ[j, k, l] * ϵ̂ᵢ[l]
        end

        Hᵢmle = ∂gᵢ' * Σinvs[i] * ∂gᵢ

        Hmle[
            slice(τᵢ, measured_components, zdim),
            slice(τᵢ, measured_components, zdim)
        ] = sparse(Hᵢmle)

        ∇ᵢmle = ΔY.ys[i]' * Σinvs[i] * ∂gᵢ

        if i == M
            Hᵢmle *= Qyf
            ∇ᵢmle *= Qyf
        else
            Hᵢmle *= Qy
            ∇ᵢmle *= Qy
        end

        ∇[slice(τᵢ, measured_components, zdim)] = ∇ᵢmle
    end

    return Hmle, ∇
end



"""
    build_mle_hessian_and_gradient_new()

Builds the hessian matrix and gradient vector for the MLE objective term.
"""
@inline function build_mle_hessian_and_gradient_new(
    ΔY::MeasurementData,
    ∂g::Function,
    ∂²g::Function,
    Qy::Float64,
    Qyf::Float64,
    Σinvs::Vector{<:AbstractMatrix},
    dims::NamedTuple,
    measured_components::AbstractVector{Int},
    T::Int, # number of time steps
    correction_term::Bool
)
    M = length(ΔY.τs)
    zdim = dims.states + dims.controls

    Hmle = spzeros(zdim * T, zdim * T)

    ∇ = zeros(zdim * T)

    for i = 1:M

        τᵢ = ΔY.τs[i]
        yᵢ = ΔY.ys[i]

        ∂gᵢ = ∂g(yᵢ)

        if correction_term
            ∂²gᵢ = ∂²g(yᵢ)
            ϵ̂ᵢ = pinv(∂gᵢ) * yᵢ
            @einsum ∂gᵢ[j, k] += ∂²gᵢ[j, k, l] * ϵ̂ᵢ[l]
        end

        Hᵢmle = ∂gᵢ' * Σinvs[i] * ∂gᵢ

        Hmle[
            slice(τᵢ, measured_components, zdim),
            slice(τᵢ, measured_components, zdim)
        ] = sparse(Hᵢmle)

        ∇ᵢmle = yᵢ' * Σinvs[i] * ∂gᵢ

        if i == M
            Hᵢmle *= Qyf
            ∇ᵢmle *= Qyf
        else
            Hᵢmle *= Qy
            ∇ᵢmle *= Qy
        end

        ∇[slice(τᵢ, measured_components, zdim)] = ∇ᵢmle
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


"""
    build_initial_and_final_constraint_matrix(Z::Traj)

Builds and vcats the matrices to pick out the initial and final states of constrained components.
"""
@inline function build_initial_and_final_constraint_matrix(Z::Traj)

    initCs = []

    for key ∈ keys(Z.initial)
        # ignore the wavefunction state constraints
        if key == :ψ̃
            continue
        end
        C = spzeros(Z.dims[key], Z.dim * Z.T)
        C[
            1:Z.dims[key],
            slice(1, Z.components[key], Z.dim)
        ] = sparse(I(Z.dims[key]))
        push!(initCs, C)
    end

    finalCs = []

    for key ∈ keys(Z.final)
        if key == :ψ̃
            continue
        end
        C = spzeros(Z.dims[key], Z.dim * Z.T)
        C[
            1:Z.dims[key],
            slice(Z.T, Z.components[key], Z.dim)
        ] = sparse(I(Z.dims[key]))
        push!(finalCs, C)
    end

    Cs = [initCs; finalCs]

    if isempty(Cs)
        return spzeros(Float64, 0, Z.dim * Z.T)
    else
        return sparse_vcat(Cs...)
    end
end

"""
    build_initial_and_final_constraints(Z::Traj)

Build the constraint vector for the initial and final states of constrained components.
"""
@inline function build_initial_and_final_constraints(Z::Traj)
    if !isempty(keys(Z.initial))
        initial_cons = vcat([Z.initial[key] for key ∈ keys(Z.initial) if key != :ψ̃]...)
        # zeros(sum([Z.dims[key] for key ∈ keys(Z.initial) if key != :ψ̃]))
    else
        initial_cons = zeros(0)
    end
    if !isempty(keys(Z.final))
        final_cons = vcat([Z.final[key] for key ∈ keys(Z.final) if key != :ψ̃]...)
        # final_cons = zeros(sum([Z.dims[key] for key ∈ keys(Z.final) if key != :ψ̃]))
    else
        final_cons = zeros(0)
    end
    return vcat(initial_cons, final_cons)
end



"""
    build_bounds_constraint_matrix(Z::Traj)

Builds the bounds matrix by vcating the matrices that pick out the states/controls that are bounded in `Z`.

Acts on all timeslices: `t ∈ 1:T`
"""
@inline function build_bounds_constraint_matrix(Z::Traj)
    Cs = []
    for key ∈ keys(Z.bounds)
        C = spzeros(Z.dims[key] * Z.T, Z.dim * Z.T)
        for t = 1:Z.T
            C[
                slice(t, Z.dims[key]),
                slice(t, Z.components[key], Z.dim)
            ] = sparse(I(Z.dims[key]))
        end
        push!(Cs, C)
    end
    if isempty(Cs)
        return spzeros(Float64, 0, Z.dim * Z.T)
    else
        return sparse_vcat(Cs...)
    end
end

"""
    build_bounds_constraints(Z::Traj)

Builds the bounds vector by vcating the bounds vectors for each bounded componenent in `Z`.

Acts on all timeslices: `t ∈ 1:T`
"""
@inline function build_bounds_constraints(Z::Traj)
    b_lbs = []
    b_ubs = []
    for key ∈ keys(Z.bounds)

        b_lb = -vcat(fill(Z.bounds[key], Z.T)...) - vec(Z[key][:, 1:Z.T])

        b_ub = vcat(fill(Z.bounds[key], Z.T)...) - vec(Z[key][:, 1:Z.T])

        push!(b_lbs, b_lb)
        push!(b_ubs, b_ub)
    end
    if isempty(b_lbs)
        return zeros(0), zeros(0)
    else
        return vcat(b_lbs...), vcat(b_ubs...)
    end
end


"""
    build_constraints(Z::Traj)

Builds the entire constraint vector -- i.e., dynamics, bounds, and initial/final constraints.
"""
@inline function build_constraints(Z::Traj)
    ∂F_cons = zeros(Z.dims.states * (Z.T - 1))
    B_lb, B_ub = build_bounds_constraints(Z)
    C_init_final = build_initial_and_final_constraints(Z)
    return vcat(∂F_cons, B_lb, C_init_final), vcat(∂F_cons, B_ub, C_init_final)
end



@inline function build_controls_constraint_matrix(
    dims::NamedTuple;
    d2u=false
)

    C_u = spzeros(
        dims.u * dims.T,
        dims.z * dims.T
    )
    if d2u
        for t = 1:dims.T
            C_u[
                slice(
                    t,
                    dims.u
                ),
                slice(
                    t,
                    dims.x - 2*dims.u + 1,
                    dims.x - dims.u,
                    dims.z
                )
            ] = sparse(I(dims.u))
        end
    else
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
    end
    return C_u
end

@inline function build_controls_constraint_bounds(
    Ẑ::Trajectory,
    u_bounds::Vector,
    dims::NamedTuple;
    d2u=false
)
    if d2u
        a_inds = (dims.x - 2*dims.u + 1):(dims.x - dims.u)

        C_u_lb = -foldr(vcat, fill(u_bounds, dims.T - 2)) -
        vcat([Ẑ.states[t][a_inds] for t = 2:dims.T - 1]...)

        C_u_ub = foldr(vcat, fill(u_bounds, dims.T - 2)) -
        vcat([Ẑ.states[t][a_inds] for t = 2:dims.T - 1]...)

        C_u_lb = [zeros(dims.u); C_u_lb; zeros(dims.u)]

        C_u_ub = [zeros(dims.u); C_u_ub; zeros(dims.u)]
    else
        C_u_lb = -foldr(vcat, fill(u_bounds, dims.T - 2)) -
            vcat(Ẑ.actions[2:dims.T - 1]...)

        C_u_ub = foldr(vcat, fill(u_bounds, dims.T - 2)) -
            vcat(Ẑ.actions[2:dims.T - 1]...)

        C_u_lb = [zeros(dims.u); C_u_lb; zeros(dims.u)]

        C_u_ub = [zeros(dims.u); C_u_ub; zeros(dims.u)]
    end
    return C_u_lb, C_u_ub
end

@inline function build_d2u_constraint_matrix(
    dims::NamedTuple
)
    C_d2u = spzeros(
        dims.u * dims.T,
        dims.z * dims.T
    )
    for t = 1:dims.T
        C_d2u[
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
    return C_d2u
end


@inline function build_d2u_constraint_bounds(
    Ẑ::Trajectory,
    dims::NamedTuple;
    d2u_bounds = fill(1e-5, dims.u)
)
    C_d2u_lb = -foldr(vcat, fill(d2u_bounds, dims.T)) -
    vcat(Ẑ.actions[1:dims.T]...)

    C_d2u_ub = foldr(vcat, fill(d2u_bounds, dims.T)) -
    vcat(Ẑ.actions[1:dims.T]...)

    return C_d2u_lb, C_d2u_ub
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

"""
    build_dynamics_constraint_jacobian(Ẑ::Traj, ∂f::Function)

Builds the Jacobian of the dynamics constraint for the given trajectory
"""
@inline function build_dynamics_constraint_jacobian(
    Ẑ::Traj,
    ∂f::Function
)
    ∂F = spzeros(
        Ẑ.dims.states * (Ẑ.T - 1),
        Ẑ.dim * Ẑ.T
    )

    for t = 1:Ẑ.T - 1

        zₜzₜ₊₁ = vec(Ẑ.data[:, t:t+1])

        ∂F[
            slice(t, Ẑ.dims.states),
            slice(t:t+1, Ẑ.dim)
        ] = sparse(∂f(zₜzₜ₊₁))
    end

    return ∂F
end





end
