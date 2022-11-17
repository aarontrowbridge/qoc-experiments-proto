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

function get_mat_iso(mat)
    mat_r = real(mat)
    mat_i = imag(mat)
    return vcat(hcat(mat_r, -mat_i),
                hcat(mat_i,  mat_r))
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

struct QuantumExperiment
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
    ydim::Int,
    τs::AbstractVector{Int};
    G_error_term=zeros(size(sys.G_drift)),
    integrator=exp
)
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
    U::Vector{Vector{Float64}}
)::MeasurementData

    T = length(U)

    Ψ̃ = Vector{typeof(experiment.ψ̃₁)}(undef, T)
    Ψ̃[1] = experiment.ψ̃₁
    CHI = 2π * -0.5469e-3
    cavity_levels = 15
    TRANSMON_E = [zeros(1); 1; zeros(2 - 2)]
    for t = 2:T
        Gₜ = Integrators.G(
            U[t - 1],
            experiment.G_drift,
            experiment.G_drives
        ) + experiment.G_error_term

        Δt = experiment.ts[t] - experiment.ts[t - 1]

        Ψ̃[t] = experiment.integrator(Gₜ * Δt) * Ψ̃[t - 1]
    end

    Ȳ = measure(
        Ψ̃,
        experiment.g,
        experiment.τs,
        experiment.ydim
    )
    println(abs2.(iso_to_ket(Ȳ.ys[end])))
    return Ȳ
end


# TODO:
# - make more general for arbitrary dynamics
#   - make work with augmented state
# - add functionality to store non updating jacobians
# - OSQP error handling

struct QuadraticProblem
    ∂f::Function
    ∂g::Function
    ∂²g::Function
    Q::Float64
    R::Float64
    Σinv::Union{Matrix{Float64}, Nothing}
    u_bounds::Vector{Float64}
    correction_term::Bool
    settings::Dict
    dims::NamedTuple
    mle::Bool
end

function QuadraticProblem(
    f::Function,
    g::Function,
    Q::Float64,
    R::Float64,
    Σ::AbstractMatrix{Float64},
    u_bounds::Vector{Float64},
    correction_term::Bool,
    settings::Dict,
    dims::NamedTuple
)
    @assert size(Σ, 1) == size(Σ, 2) == dims.y

    ∂f(zz) = ForwardDiff.jacobian(f, zz)

    ∂g(x) = ForwardDiff.jacobian(g, x)

    function ∂²g(x)
        H = ForwardDiff.jacobian(u -> vec(∂g(u)), x)
        return reshape(H, dims.y, dims.x, dims.x)
    end

    return QuadraticProblem(
        ∂f,
        ∂g,
        ∂²g,
        Q,
        R,
        inv(Σ),
        u_bounds,
        correction_term,
        settings,
        dims,
        !isnothing(Σ)
    )
end

function (QP::QuadraticProblem)(
    Ẑ::Trajectory,
    ΔY::MeasurementData,
)
    model = OSQP.Model()

    if QP.mle
        H, ∇ = build_mle_hessian_and_gradient(QP, Ẑ, ΔY)
        A, lb, ub = build_constraint_matrix(QP, Ẑ, ΔY)
    else
        H = build_regularization_hessian(QP)
        A, lb, ub = build_constraint_matrix(QP, Ẑ, ΔY)
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
    QP::QuadraticProblem
)
    Hₜ = spdiagm([QP.Q * ones(QP.dims.x); QP.R * ones(QP.dims.u)])
    H = kron(sparse(I(QP.dims.T)), sparse(Hₜ))
    return H
end

@inline function build_mle_hessian_and_gradient(
    QP::QuadraticProblem,
    Ẑ::Trajectory,
    ΔY::MeasurementData
)
    Hreg = build_regularization_hessian(QP)

    Hmle = spzeros(QP.dims.z * QP.dims.T, QP.dims.z * QP.dims.T)

    ∇ = zeros(QP.dims.z * QP.dims.T)

    for i = 1:QP.dims.M

        τᵢ = ΔY.times[i]

        ∂gᵢ = QP.∂g(Ẑ.states[τᵢ])

        if QP.correction_term
            ∂²gᵢ = QP.∂²g(Ẑ.states[τᵢ])
            ϵ̂ᵢ = pinv(∂gᵢ) * ΔY.ys[i]
            @einsum ∂gᵢ[j, k] += ∂²gᵢ[j, k, l] * ϵ̂ᵢ[l]
        end

        Hᵢmle = ∂gᵢ' * QP.Σinv * ∂gᵢ

        Hmle[
            slice(τᵢ, QP.dims.x, QP.dims.z),
            slice(τᵢ, QP.dims.x, QP.dims.z)
        ] = sparse(Hᵢmle)

        ∇ᵢmle = ΔY.ys[i]' * QP.Σinv * ∂gᵢ

        ∇[slice(τᵢ, QP.dims.x, QP.dims.z)] = ∇ᵢmle
    end

    H = Hreg + Hmle

    return H, ∇
end




@inline function build_constraint_matrix(
    QP::QuadraticProblem,
    Ẑ::Trajectory,
    ΔY::MeasurementData,
)
    ∂F = build_dynamics_constraint_jacobian(QP, Ẑ)

    f_cons = zeros(QP.dims.x * (QP.dims.T - 1))

    C = build_constrols_constraint_matrix(QP)

    u_lb = -foldr(vcat, fill(QP.u_bounds, Ẑ.T - 2)) -
        vcat(Ẑ.actions[2:Ẑ.T - 1]...)

    u_ub = foldr(vcat, fill(QP.u_bounds, Ẑ.T - 2)) -
        vcat(Ẑ.actions[2:Ẑ.T - 1]...)

    # constrain state matrix for state at time 1
    C_x₁ = sparse_hcat(
        I(QP.dims.x),
        spzeros(QP.dims.x, QP.dims.u + QP.dims.z * (QP.dims.T - 1))
    )

    if QP.mle
        A = sparse_vcat(∂F, C, C_x₁)
        lb = vcat(
            f_cons,
            zeros(QP.dims.u),
            u_lb,
            zeros(QP.dims.u),
            zeros(QP.dims.x)
        )
        ub = vcat(
            f_cons,
            zeros(QP.dims.u),
            u_ub,
            zeros(QP.dims.u),
            zeros(QP.dims.x)
        )
    else
        ∂G = build_measurement_constraint_jacobian(
            QP,
            Ẑ,
            ΔY
        )
        A = sparse_vcat(∂F, ∂G, C, C_x₁)
        g_cons = -vcat(ΔY.ys...)
        lb = vcat(
            f_cons,
            g_cons,
            zeroes(QP.dims.u),
            u_lb,
            zeros(QP.dims.u),
            zeros(QP.dims.x)
        )
        ub = vcat(
            f_cons,
            g_cons,
            zeroes(QP.dims.u),
            u_ub,
            zeros(QP.dims.u),
            zeros(QP.dims.x)
        )
    end

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
    ∂G = spzeros(QP.dims.y * QP.dims.M, QP.dims.z * QP.dims.T)

    for i = 1:QP.dims.M

        τᵢ = ΔY.times[i]

        ∂gᵢ = QP.∂g(Ẑ.states[τᵢ])

        if QP.correction_term
            ∂²gᵢ = QP.∂²g(Ẑ.states[τᵢ])
            ϵ̂ᵢ = pinv(∂gᵢ) * ΔY.ys[i]
            @einsum ∂gᵢ[j, k] += ∂²gᵢ[j, k, l] * ϵ̂ᵢ[l]
        end

        ∂G[
            slice(i, QP.dims.y),
            slice(τᵢ, QP.dims.x, QP.dims.z)
        ] = sparse(∂gᵢ)
    end

    return ∂G
end

# TODO: add feat to just store jacobian of goal traj
@inline function build_dynamics_constraint_jacobian(
    QP::QuadraticProblem,
    Ẑ::Trajectory
)
    ∂F = spzeros(
        QP.dims.x * (QP.dims.T - 1),
        QP.dims.z * QP.dims.T
    )

    for t = 1:QP.dims.T - 1

        zₜ = [
            Ẑ.states[t];
            Ẑ.actions[t];
            Ẑ.states[t + 1];
            Ẑ.actions[t + 1]
        ]

        ∂F[
            slice(t, QP.dims.x),
            slice(t, QP.dims.z; stretch=QP.dims.z)
        ] = sparse(QP.∂f(zₜ))
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
    experiment::QuantumExperiment
    settings::Dict

    function ILCProblem(
        sys::QuantumSystem,
        Ẑgoal::Trajectory,
        experiment::QuantumExperiment;
        integrator=:FourthOrderPade,
        Q=1.0,
        R=1.0,
        # identity matrix of Float64
        Σ=Diagonal(fill(1.0, experiment.ydim)),
        u_bounds=sys.a_bounds,
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

        dynamics = eval(integrator)(sys)

        dims = (
            x=size(Ẑgoal.states[1], 1),
            u=size(Ẑgoal.actions[1], 1),
            z=size(Ẑgoal.states[1], 1) + size(Ẑgoal.actions[1], 1),
            y=experiment.ydim,
            T=Ẑgoal.T,
            M=length(experiment.τs)
        )

        f = zz -> begin
            xₜ₊₁ = zz[dims.z .+ (1:dims.x)]
            xₜ = zz[1:dims.x]
            uₜ = zz[dims.x .+ (1:dims.u)]
            return dynamics(xₜ₊₁, xₜ, uₜ, Ẑgoal.Δt)
        end

        QP = QuadraticProblem(
            f, experiment.g,
            Q, R, Σ,
            u_bounds,
            correction_term,
            QP_settings,
            dims
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
    k = 1
    while norm(ΔY, prob.settings[:norm_p]) > prob.settings[:tol]
        if k > prob.settings[:max_iter]
            @info "max iterations reached" max_iter = prob.settings[:max_iter] "|ΔY|" = norm(ΔY, prob.settings[:norm_p]) tol = prob.settings[:tol]
            return
        end
        # TODO: make jacobians constant about nominal trajectory
        if prob.settings[:verbose]
            println("iter = ", k)
            println("|ΔY| =      ", norm(ΔY, prob.settings[:norm_p]))
            println(
                "fidelity =  ",
                fidelity(prob.Ẑ.states[end], prob.Ẑgoal.states[end])
            )
            println()
        end
        ΔZ = prob.QP(prob.Ẑ, ΔY)
        prob.Ẑ = prob.Ẑ + ΔZ
        println("norm(ψ₁) = ", norm(prob.Ẑ.states[1]))
        U = prob.Ẑ.actions
        Ȳ = prob.experiment(U)
        ΔYnext = Ȳ - prob.Ygoal
        if norm(ΔYnext, prob.settings[:norm_p]) > norm(ΔY, prob.settings[:norm_p])
            println("   backtracking")
            println()
            i = 1
        end
        while norm(ΔYnext, prob.settings[:norm_p]) > norm(ΔY, prob.settings[:norm_p])
            if i > 10
                println("   max backtracking iterations reached")
                ΔY = ΔYnext
                return
            end
            prob.Ẑ = prob.Ẑ - ΔZ
            ΔZ = 0.5 * ΔZ
            prob.Ẑ = prob.Ẑ + ΔZ
            U = prob.Ẑ.actions
            Ȳ = prob.experiment(U)
            ΔYnext = Ȳ - prob.Ygoal
            println("       iter = $i")
            println("       |ΔY| = ", norm(ΔYnext, prob.settings[:norm_p]))
            println()
            i += 1
        end
        ΔY = ΔYnext
        push!(prob.Ȳs, Ȳ)
        push!(prob.Us, hcat(U...))
        k += 1
    end
    @info "ILC converged!" "|ΔY|" = norm(ΔY, prob.settings[:norm_p]) tol = prob.settings[:tol] iter = k
end



end
