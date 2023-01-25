module ILCExperiments

export MeasurementData
export measure

export AbstractExperiment
export HardwareExperiment
export QuantumExperiment
export experiment
export rollout
export nominal_rollout

using ..Trajectories
using ..ILCTrajectories
using ..Integrators
using ..QuantumSystems

using LinearAlgebra

struct MeasurementData
    ys::Vector{<:AbstractVector{Float64}}
    Σs::Vector{<:AbstractMatrix{Float64}}
    τs::AbstractVector{Int}
    ydim::Int
end

function MeasurementData(
    ys::Vector{<:AbstractVector{Float64}},
    τs::AbstractVector{Int},
    ydim::Int,
    σ::Float64=1.0
)
    return MeasurementData(ys, fill(σ * I(ydim), length(ys)), τs, ydim)
end

function Base.:-(
    data1::MeasurementData,
    data2::MeasurementData
)
    @assert data1.ydim == data2.ydim
    @assert data1.τs == data2.τs
    return MeasurementData(
        data1.ys .- data2.ys,
        data1.Σs, #TODO: reformulate for correctness -- currently not used in calculations
        data1.τs,
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
    ydim::Int;
    Σ=nothing
)::MeasurementData
    @assert size(g(Z.states[1]), 1) == ydim
    ys = Vector{Vector{Float64}}(undef, length(τs))
    for (i, τ) in enumerate(τs)
        ys[i] = g(Z.states[τ])
    end
    return MeasurementData(ys, τs, ydim)
end

function measure(
    Ψ̃::AbstractMatrix{Float64},
    g::Function,
    τs::AbstractVector{Int},
    ydim::Int;
    σ::Float64=1.0
)::MeasurementData
    @assert length(g(Ψ̃[:, 1])) == ydim

    ys = Vector{Vector{Float64}}(undef, length(τs))

    for (i, τ) in enumerate(τs)
        ys[i] = g(Ψ̃[:, τ])
    end

    return MeasurementData(ys, τs, ydim, σ)
end

function measure_noisy(
    Ψ̃::AbstractMatrix{Float64},
    g::Function,
    τs::AbstractVector{Int},
    ydim::Int
)::MeasurementData
    @assert length(g(Ψ̃[:, 1])) == ydim

    ys = Vector{Vector{Float64}}(undef, length(τs))
    Σs = Vector{<:AbstractMatrix{Float64}}(undef, length(τs))

    for (i, τ) in enumerate(τs)
        y, Σ = g(Ψ̃[:, τ])
        ys[i] = y
        Σs[i] = Σ
    end

    return MeasurementData(ys, Σs, τs, ydim)
end


abstract type AbstractExperiment end

struct HardwareExperiment <: AbstractExperiment
    g_hardware::Function
    g_analytic::Function
    τs::AbstractVector{Int}
    ydim::Int
    M::Int

    function HardwareExperiment(
        g_hardware::Function,
        g_analytic::Function,
        τs::AbstractVector{Int},
        ydim::Int
    )
        return new(g_hardware, g_analytic, τs, ydim, length(τs))
    end
end

function (experiment::HardwareExperiment)(
    us::Vector{Vector{Float64}},
    times::AbstractVector{Float64};
    backtracking=false
)::MeasurementData
    if !backtracking
        return experiment.g_hardware(us, times, experiment.τs)
    else
        return experiment.g_hardware(us, times, [experiment.τs[end]])
    end
end

struct QuantumExperiment <: AbstractExperiment
    ψ̃₁::Vector{Float64}
    g::Function
    ydim::Int
    τs::AbstractVector{Int}
    M::Int
    integrator::Function
    control_transform::Function
    G_drift::AbstractMatrix{Float64}
    G_drives::Vector{AbstractMatrix{Float64}}
    G_error_term::AbstractMatrix{Float64}
end

function QuantumExperiment(
    sys::QuantumSystem,
    ψ̃₁::AbstractVector{Float64},
    g::Function,
    τs::AbstractVector{Int};
    G_error_term=zeros(size(sys.G_drift)),
    integrator=exp,
    control_transform=u->u
)
    ydim = size(g(ψ̃₁), 1)
    return QuantumExperiment(
        ψ̃₁,
        g,
        ydim,
        τs,
        length(τs),
        integrator,
        control_transform,
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
    A::AbstractMatrix{Float64},
    times::AbstractVector{Float64};
    backtracking=false,
    σ=1.0
)::MeasurementData

    Ψ̃ = rollout(A, times, experiment)

    if backtracking
        Ȳ = measure(
            Ψ̃,
            experiment.g,
            experiment.τs[end:end],
            experiment.ydim;
            σ=σ
        )
    else
        Ȳ = measure(
            Ψ̃,
            experiment.g,
            experiment.τs,
            experiment.ydim;
            σ=σ
        )
    end

    return Ȳ
end

function rollout(
    A::AbstractMatrix{Float64},
    times::AbstractVector{Float64},
    experiment::QuantumExperiment
)
    T = size(A, 2)

    Ψ̃ = zeros(eltype(experiment.ψ̃₁), length(experiment.ψ̃₁), T)

    Ψ̃[:, 1] .= experiment.ψ̃₁

    Â = experiment.control_transform(A)

    for t = 2:T
        âₜ₋₁ = Â[:, t - 1]
        Gₜ = Integrators.G(
            âₜ₋₁,
            experiment.G_drift,
            experiment.G_drives
        ) + experiment.G_error_term

        Δt = times[t] - times[t - 1]

        Ψ̃[:, t] .= experiment.integrator(Gₜ * Δt) * Ψ̃[:, t - 1]
    end

    return Ψ̃
end

function nominal_rollout(
    A::AbstractMatrix{Float64},
    times::AbstractVector{Float64},
    experiment::QuantumExperiment;
    integrator=Integrators.fourth_order_pade
)
    T = size(A, 2)

    Ψ̃ = zeros(eltype(experiment.ψ̃₁), length(experiment.ψ̃₁), T)

    Ψ̃[:, 1] .= experiment.ψ̃₁

    for t = 2:T
        aₜ₋₁ = A[:, t - 1]
        Gₜ = Integrators.G(
            aₜ₋₁,
            experiment.G_drift,
            experiment.G_drives
        )

        Δt = times[t] - times[t - 1]

        Ψ̃[:, t] .= integrator(Gₜ * Δt) * Ψ̃[:, t - 1]
    end

    return Ψ̃
end






end
