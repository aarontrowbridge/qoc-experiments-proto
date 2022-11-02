module Dynamics

export AbstractDynamics
export QuantumDynamics
export MinTimeQuantumDynamics

using ..Utils
using ..QuantumLogic
using ..QuantumSystems
using ..Integrators

using LinearAlgebra
using SparseArrays

#
# dynamics functions
#

@views function dynamics(
    xₜ₊₁::AbstractVector{R},
    xₜ::AbstractVector{R},
    uₜ::AbstractVector{R},
    Δt::Real,
    P::QuantumIntegrator,
    sys::QuantumSystemSystem
)::Vector{R} where R <: Real
    augsₜ = xₜ[(sys.n_wfn_states + 1):end]
    augsₜ₊₁ = xₜ₊₁[(sys.n_wfn_states + 1):end]
    controlsₜ = [augsₜ[(sys.ncontrols + 1):end]; uₜ]
    δaugs = augsₜ₊₁ - (augsₜ + controlsₜ * Δt)
    δψ̃s = zeros(typeof(xₜ[1]), sys.n_wfn_states)
    aₜ = augsₜ[slice(1 + sys.∫a, sys.ncontrols)]
    for i = 1:sys.nqstates
        ψ̃ⁱ_slice = slice(i, sys.isodim)
        ψ̃ⁱₜ = xₜ[ψ̃ⁱ_slice]
        ψ̃ⁱₜ₊₁ = xₜ₊₁[ψ̃ⁱ_slice]
        δψ̃s[ψ̃ⁱslice] = P(ψ̃ⁱₜ₊₁, ψ̃ⁱₜ, aₜ, Δt)
    end
    return [δψ̃s; δaugs]
end

@views function fₜ(
    zₜ::AbstractVector{R},
    zₜ₊₁::AbstractVector{R},
    Δt::Real,
    P::QuantumIntegrator,
    sys::QuantumSystem
)::Vector{R} where R <: Real
    xₜ₊₁ = zₜ₊₁[1:sys.nstates]
    xₜ = zₜ[1:sys.nstates]
    uₜ = zₜ[
        sys.n_wfn_states .+
        slice(sys.augdim + 1, sys.ncontrols)
    ]
    return dynamics(xₜ₊₁, xₜ, uₜ, Δt, P, sys)
end

@views function F(
    Z::AbstractVector{R},
    Δt::Real,
    T::Int,
    P::QuantumIntegrator,
    sys::QuantumSystem
)::Vector{R} where R <: Real
    δX = zeros(sys.nstates * (T - 1))
    for t = 1:T-1
        zₜ = Z[slice(t, sys.vardim)]
        zₜ₊₁ = Z[slice(t + 1, sys.vardim)]
        δX[slice(t, sys.nstates)] = fₜ(zₜ, zₜ₊₁, Δt, P, sys)
    end
    return δX
end

@views function F(
    Z::AbstractVector{R},
    Δt::AbstractVector{R},
    T::Int,
    P::QuantumIntegrator,
    sys::QuantumSystem
)::Vector{R} where R <: Real
    δX = zeros(sys.nstates * (T - 1))
    for t = 1:T-1
        zₜ = Z[slice(t, sys.vardim)]
        zₜ₊₁ = Z[slice(t + 1, sys.vardim)]
        δX[slice(t, sys.nstates)] = fₜ(zₜ, zₜ₊₁, Δt, P, sys)
    end
    return δX
end




@views function ∂zₜfₜ(
    zₜ::AbstractVector{R},
    zₜ₊₁::AbstractVector{R},
    Δt::Real,
    P::QuantumIntegrator,
    sys::QuantumSystem
)::Vector{R} where R <: Real

    ∂s = []

    # ∂ψ̃ⁱₜPⁱ blocks
    aₜ = zₜ[sys.n_wfn_states .+ slice(1 + sys.∫a, sys.ncontrols)]
    ∂ψ̃ⁱₜPⁱ = ∂ψ̃ⁱₜ(P, aₜ, Δt)
    for i = 1:sys.nqstates
        append!(∂s, ∂ψ̃ⁱₜPⁱ)
    end

    # ∂aₜPⁱ blocks
    for i = 1:sys.nqstates
        ψ̃ⁱ_slice = slice(i, sys.isodim)
        ψ̃ⁱₜ₊₁ = zₜ₊₁[ψ̃ⁱ_slice]
        ψ̃ⁱₜ = zₜ[ψ̃ⁱ_slice]
        ∂aₜPⁱ = ∂aₜ(P, ψ̃ⁱₜ₊₁, ψ̃ⁱₜ, aₜ, Δt)
        append!(∂s, ∂aₜPⁱ)
    end

    # -I blocks on main diagonal
    for _ = 1:sys.n_aug_states
        append!(∂s, -1.0)
    end

    # -Δt⋅I blocks on shifted diagonal
    for _ = 1:sys.n_aug_states
        append!(∂s, -Δt)
    end

    return ∂s
end

function ∂zₜfₜstructure(
    sys::QuantumSystem
)::Vector{Tuple{Int, Int}}

    structure = []

    # ∂ψ̃ⁱₜPⁱ blocks
    for i = 1:sys.nqstates
        for j = 1:sys.isodim     # jth column of ∂ψ̃ⁱₜPⁱ
            for k = 1:sys.isodim # kth row
                kj = (
                    index(i, k, sys.isodim),
                    index(i, j, sys.isodim)
                )
                push!(structure, kj)
            end
        end
    end

    # ∂aₜPⁱ blocks
    for i = 1:sys.nqstates
        for j = 1:sys.ncontrols  # jth column: ∂aʲₜPⁱ
            for k = 1:sys.isodim # kth row of ψ̃ⁱᵏₜ
                kj = (
                    index(i, k, sys.isodim),
                    sys.n_wfn_states + sys.∫a * sys.ncontrols + j
                )
                push!(structure, kj)
            end
        end
    end

    # -I blocks on main diagonal
    for k = 1:sys.n_aug_states
        kk = (k, k) .+ sys.n_wfn_states
        push!(structure, kk)
    end

    # -Δt⋅I blocks on shifted diagonal
    for k = 1:sys.n_aug_states
        kk_shifted = (k, k + sys.ncontrols) .+ sys.n_wfn_states
        push!(structure, kk_shifted)
    end

    return structure
end




@views function ∂zₜ₊₁fₜ(
    zₜ::AbstractVector{R},
    Δt::Real,
    P::QuantumIntegrator,
    sys::QuantumSystem
)::Vector{R} where R <: Real

    ∂s = []

    # ∂ψ̃ⁱₜPⁱ blocks
    aₜ = zₜ[sys.n_wfn_states .+ slice(1 + sys.∫a, sys.ncontrols)]
    ∂ψ̃ⁱₜ₊₁Pⁱ = ∂ψ̃ⁱₜ₊₁(P, aₜ, Δt)
    for i = 1:sys.nqstates
        append!(∂s, ∂ψ̃ⁱₜ₊₁Pⁱ)
    end

    # I for controls on main diagonal
    for _ = 1:sys.n_aug_states
        append!(∂s, 1.0)
    end

    return ∂s
end

function ∂zₜ₊₁fₜstructure(
    sys::QuantumSystem
)::Vector{Tuple{Int, Int}}

    structure = []

    # ∂ψ̃ⁱₜPⁱ blocks
    for i = 1:sys.nqstates
        for j = 1:sys.isodim     # jth column: ∂ψ̃ⁱʲₜPⁱ
            for k = 1:sys.isodim # kth row: ∂ψ̃ⁱʲₜPⁱᵏ
                kj = (
                    index(i, k, sys.isodim),
                    index(i, j, sys.isodim)
                )
                push!(structure, kj)
            end
        end
    end

    # I for controls on main diagonal
    for k = 1:sys.n_aug_states
        kk = sys.n_wfn_states .+ (k, k)
        push!(structure, kk)
    end
end

@views function ∂Δtₜfₜ(
    zₜ::AbstractVector{R},
    zₜ₊₁::AbstractVector{R},
    Δtₜ::Real,
    P::QuantumIntegrator,
    sys::QuantumSystem
)::Vector{R} where R <: Real

    ∂s = []

    aₜ = zₜ[sys.n_wfn_states .+ slice(1 + sys.∫a, sys.ncontrols)]

    # ψ̃ⁱ blocks
    for i = 1:sys.nqstates
        ψ̃ⁱ_slice = slice(i, sys.isodim)
        ψ̃ⁱₜ₊₁ = zₜ₊₁[ψ̃ⁱ_slice]
        ψ̃ⁱₜ = zₜ[ψ̃ⁱ_slice]
        ∂ΔtₜPⁱ = ∂Δtₜ(P, ψ̃ⁱₜ₊₁, ψ̃ⁱₜ, aₜ, Δtₜ)
        append!(∂s, ∂ΔtₜPⁱ)
    end

    # ∫a block
    if sys.∫a
        append!(∂s, -aₜ)
    end

    # controls blocks (e.g. ȧₜ, uₜ if control_order == 2)
    for n = 1:sys.control_order
        nth_order_controls =
            zₜ[sys.n_wfn_states .+ slice(sys.∫a + 1 + n, sys.ncontrols)]
        append!(∂s, -nth_order_controls)
    end

    return ∂s
end

function ∂Δtₜfₜstructure(
    sys::QuantumSystem
)::Vector{Int}
    return collect(1:sys.nstates)
end


@views function ∂F(
    Z::AbstractVector{R},
    Δt::Real,
    T::Int,
    P::QuantumIntegrator,
    sys::QuantumSystem
)::Vector{R} where R <: Real
    ∂s = []
    for t = 1:T-1
        zₜ = Z[slice(t, sys.vardim)]
        zₜ₊₁ = Z[slice(t + 1, sys.vardim)]
        append!(∂s, ∂zₜfₜ(zₜ, zₜ₊₁, Δt, P, sys))
        append!(∂s, ∂zₜ₊₁fₜ(zₜ, Δt, P, sys))
    end
    return ∂s
end

function ∂F_structure(
    sys::QuantumSystem,
    T::Int
)::Vector{Tuple{Int, Int}}

    structure = []

    for t = 1:T-1

        for (k, j) in ∂zₜfₜstructure(sys)
            kₜ = k + index(t, 0, sys.nstates)
            jₜ = j + index(t, 0, sys.vardim)
            push!(structure, (kₜ, jₜ))
        end

        for (k, j) in ∂zₜ₊₁fₜstructure(sys)
            kₜ = k + index(t, 0, sys.nstates)
            jₜ = j + index(t + 1, 0, sys.vardim)
            push!(structure, (kₜ, jₜ))
        end
    end

    return structure
end

# mintime version
@views function ∂F(
    Z::AbstractVector{R},
    Δt::AbstractVector{R},
    T::Int,
    P::QuantumIntegrator,
    sys::QuantumSystem
)::Vector{R} where R <: Real
    ∂s = []
    for t = 1:T-1
        zₜ = Z[slice(t, sys.vardim)]
        zₜ₊₁ = Z[slice(t + 1, sys.vardim)]
        Δtₜ = Δt[t]
        append!(∂s, ∂zₜfₜ(zₜ, zₜ₊₁, Δtₜ, P, sys))
        append!(∂s, ∂zₜ₊₁fₜ(zₜ, Δtₜ, P, sys))
        append!(∂s, ∂Δtₜfₜ(zₜ, zₜ₊₁, Δtₜ, P, sys))
    end
    return ∂s
end

function ∂F_structure_mintime(
    sys::QuantumSystem,
    T::Int
)::Vector{Tuple{Int, Int}}

    N = sys.vardim * T

    structure = []

    for t = 1:T-1

        for (k, j) in ∂zₜfₜstructure(sys)
            kₜ = k + index(t, 0, sys.nstates)
            jₜ = j + index(t, 0, sys.vardim)
            push!(structure, (kₜ, jₜ))
        end

        for (k, j) in ∂zₜ₊₁fₜstructure(sys)
            kₜ = k + index(t, 0, sys.nstates)
            jₜ = j + index(t + 1, 0, sys.vardim)
            push!(structure, (kₜ, jₜ))
        end

        for i in ∂Δtₜfₜstructure(sys)
            iₜ = i + index(t, 0, sys.nstates)
            jₜ = N + index(t, 0, 1)
            push!(structure, (iₜ, jₜ))
        end
    end

    return structure
end



@views function μ∂²zₜfₜ(
    μₜ::AbstractVector{R},
    zₜ::AbstractVector{R},
    zₜ₊₁::AbstractVector{R},
    Δt::Real,
    P::QuantumIntegrator,
    sys::QuantumSystem
)::Vector{R} where R <: Real

    μₜ∂²s = []

    aₜ = zₜ[sys.n_wfn_states .+ slice(1 + sys.∫a, sys.ncontrols)]
    Ψ̃ₜ = zₜ[1:sys.n_wfn_states]
    Ψ̃ₜ₊₁ = zₜ₊₁[1:sys.n_wfn_states]

    if P isa FourthOrderPade
        # μₜ∂²aₜPⁱₜ block (upper triangular portion)
        μₜ∂²aₜPⁱₜ = μₜ∂²aₜ(P, μₜ, Ψ̃ₜ₊₁, Ψ̃ₜ, aₜ, Δt)
        μₜ∂²aₜPⁱₜupper = [μₜ∂²aₜPⁱₜ[k, j] for j = 1:sys.ncontrols for k = 1:j]
        append!(μₜ∂²s, μₜ∂²aₜPⁱₜupper)
    end

    # μⁱₜ∂aₜ∂ψ̃ⁱₜPⁱₜ blocks
    for i = 1:sys.nqstates
        μⁱₜ = μₜ[slice(i, sys.isodim)]
        append!(μₜ∂²s, μⁱₜ∂aₜ∂ψ̃ⁱₜ(P, μⁱₜ, aₜ, Δt))
    end

    return μₜ∂²s
end

function μ∂²zₜfₜstructure(
    sys::QuantumSystem,
    fourth_order_pade::Bool
)::Vector{Tuple{Int, Int}}

    structure = []

    if fourth_order_pade
        # μₜ∂²aₜPⁱₜ blocks
        for j = 1:sys.ncontrols # jth column: j ∈ {1..ncontrols}
            for k = 1:j         # kth row:    k ∈ {1..j}
                kj = (sys.n_wfn_states + sys.∫a * sys.ncontrols) .+ (k, j)
                push!(structure, kj)
            end
        end
    end

    # μⁱₜ∂aₜ∂ψ̃ⁱₜPⁱₜ blocks
    for i = 1:sys.nqstates       # ith qstate: i ∈ {1..nqstates}
        for j = 1:sys.ncontrols  # jth column: j ∈ {1..ncontrols}
            for l = 1:sys.isodim # lth row:    l ∈ {1..isodim}
                lⁱj = (
                    index(i, l, sys.isodim),
                    sys.n_wfn_states + index(1 + sys.∫a, j, sys.ncontrols)
                )
                push!(structure, lⁱj)
            end
        end
    end

    return structure
end


@views function μ∂zₜ∂zₜ₊₁fₜ(
    μₜ::AbstractVector{R},
    zₜ::AbstractVector{R},
    Δt::Real,
    P::QuantumIntegrator,
    sys::QuantumSystem
)::Vector{R} where R <: Real

    μₜ∂²s = []

    aₜ = zₜ[sys.n_wfn_states .+ slice(1 + sys.∫a, sys.ncontrols)]

    # μⁱₜ∂aₜ∂ψ̃ⁱₜ₊₁Pⁱₜ blocks
    for i = 1:sys.nqstates
        μⁱₜ = μₜ[slice(i, sys.isodim)]
        append!(μₜ∂²s, μⁱₜ∂aₜ∂ψ̃ⁱₜ₊₁(P, μⁱₜ, aₜ, Δt))
    end

    return μₜ∂²s
end

function μ∂zₜ∂zₜ₊₁fₜstructure(
    sys::QuantumSystem
)::Vector{Tuple{Int, Int}}

    structure = []

    # μⁱₜ∂aₜ∂ψ̃ⁱₜPⁱₜ blocks
    for i = 1:sys.nqstates          # ith qstate: i ∈ {1..nqstates}
        for l = 1:sys.isodim        # lth column: l ∈ {1..isodim}
            for j = 1:sys.ncontrols # jth row:    j ∈ {1..ncontrols}
                jlⁱ = (
                    sys.n_wfn_states + index(1 + sys.∫a, j, sys.ncontrols),
                    index(i, l, sys.isodim)
                )
                push!(structure, jlⁱ)
            end
        end
    end

    return structure
end

@views function μₜ∂²Δtₜfₜ(
    μₜ::AbstractVector{R},
    zₜ::AbstractVector{R},
    zₜ₊₁::AbstractVector{R},
    P::QuantumIntegrator,
    sys::QuantumSystem
)::R where R <: Real
    Ψ̃ₜ₊₁ = zₜ₊₁[1:sys.n_wfn_states]
    Ψ̃ₜ = zₜ[1:sys.n_wfn_states]
    aₜ = zₜ[sys.n_wfn_states .+ slice(1 + sys.∫a, sys.ncontrols)]
    return μₜ∂²Δtₜ(P, μₜ, Ψ̃ₜ₊₁, Ψ̃ₜ, aₜ)
end

@views function μₜ∂Δtₜ∂zₜfₜ(
    μₜ::AbstractVector{R},
    zₜ::AbstractVector{R},
    zₜ₊₁::AbstractVector{R},
    Δtₜ::Real,
    P::QuantumIntegrator,
    sys::QuantumSystem
)::Vector{R} where R <: Real

    μₜ∂²s = []

    Ψ̃ₜ₊₁ = zₜ₊₁[1:sys.n_wfn_states]
    Ψ̃ₜ = zₜ[1:sys.n_wfn_states]
    aₜ = zₜ[sys.n_wfn_states .+ slice(1 + sys.∫a, sys.ncontrols)]

    # ψ̃ⁱₜ blocks
    for i = 1:sys.nqstates
        μⁱₜ = μₜ[slice(i, sys.isodim)]
        append!(μₜ∂²s, μⁱₜ∂Δtₜ∂ψ̃ⁱₜ(P, μⁱₜ, aₜ, Δtₜ))
    end

    # aₜ block (0th order control)
    μₜ∂Δtₜ∂aₜPₜ = μₜ∂Δtₜ∂aₜ(P, μₜ, Ψ̃ₜ₊₁, Ψ̃ₜ, aₜ, Δtₜ)
    if sys.∫a
        μₜ_∫a = μₜ[sys.n_wfn_states .+ slice(1, sys.ncontrols)]
        append!(μₜ∂²s, -μₜ_∫a + μₜ∂Δtₜ∂aₜPₜ)
    else
        append!(μₜ∂²s, μₜ∂Δtₜ∂aₜPₜ)
    end

    # augmented state blocks (1-nth order controls)
    for n = 1:sys.control_order
        μₜ_n = μₜ[sys.n_wfn_states .+ slice(sys.∫a + n, sys.ncontrols)]
        append!(μₜ∂²s, -μₜ_n)
    end

    return μₜ∂²s
end

@views function μₜ∂Δtₜ∂zₜ₊₁fₜ(
    μₜ::AbstractVector{R},
    zₜ::AbstractVector{R},
    Δtₜ::Real,
    P::QuantumIntegrator,
    sys::QuantumSystem
)::Vector{R} where R <: Real

    μₜ∂²s = []

    aₜ = zₜ[sys.n_wfn_states .+ slice(1 + sys.∫a, sys.ncontrols)]

    # ψ̃ⁱₜ₊₁ blocks
    for i = 1:sys.nqstates
        μⁱₜ = μₜ[slice(i, sys.isodim)]
        append!(μₜ∂²s, μⁱₜ∂Δtₜ∂ψ̃ⁱₜ₊₁(P, μⁱₜ, aₜ, Δtₜ))
    end

    return μₜ∂²s
end

@views function μ∂Δt∂F(
    μ::AbstractVector{R},
    Z::AbstractVector{R},
    Δt::AbstractVector{R},
    T::Int,
    P::QuantumIntegrator,
    sys::QuantumSystem
)::Vector{R} where R <: Real

    μ∂²s = []

    for t = 1:T
        μₜ = μ[slice(t, sys.nstates)]
        zₜ = Z[slice(t, sys.vardim)]
        zₜ₊₁ = Z[slice(t + 1, sys.vardim)]
        Δtₜ = Δt[t]
        append!(μ∂²s, μₜ∂Δtₜ∂zₜfₜ(μₜ, zₜ, zₜ₊₁, Δtₜ, P, sys))
        append!(μ∂²s, μₜ∂Δtₜ∂zₜ₊₁fₜ(μₜ, zₜ, Δtₜ, P, sys))
    end

    return μ∂²s
end

@views function μ∂²ΔtF(
    μ::AbstractVector{R},
    Z::AbstractVector{R},
    T::Int,
    P::QuantumIntegrator,
    sys::QuantumSystem
)::Vector{R} where R <: Real

    μ∂²s = []

    for t = 1:T
        μₜ = μ[slice(t, sys.nstates)]
        zₜ = Z[slice(t, sys.vardim)]
        zₜ₊₁ = Z[slice(t + 1, sys.vardim)]
        append!(μ∂²s, μₜ∂²Δtₜfₜ(μₜ, zₜ, zₜ₊₁, P, sys))
    end

    return μ∂s²
end


@views function μ∂²F(
    μ::AbstractVector{R},
    Z::AbstractVector{R},
    Δt::Real,
    T::Int,
    P::QuantumIntegrator,
    sys::QuantumSystem
)::Vector{R} where R <: Real

    μ∂²s = []

    for t = 1:T-1
        μₜ = μ[slice(t, sys.nstates)]
        zₜ = Z[slice(t, sys.vardim)]
        zₜ₊₁ = Z[slice(t + 1, sys.vardim)]
        append!(μ∂²s, μ∂²zₜfₜ(μₜ, zₜ, zₜ₊₁, Δt, P, sys))
        append!(μ∂²s, μ∂zₜ∂zₜ₊₁fₜ(μₜ, zₜ, Δt, P, sys))
    end

    return μ∂²s
end

@views function μ∂²F(
    μ::AbstractVector{R},
    Z::AbstractVector{R},
    Δt::AbstractVector{R},
    T::Int,
    P::QuantumIntegrator,
    sys::QuantumSystem
)::Vector{R} where R <: Real

    μ∂²s = []

    for t = 1:T-1
        μₜ = μ[slice(t, sys.nstates)]
        zₜ = Z[slice(t, sys.vardim)]
        zₜ₊₁ = Z[slice(t + 1, sys.vardim)]
        Δtₜ = Δt[t]
        append!(μ∂²s, μ∂²zₜfₜ(μₜ, zₜ, zₜ₊₁, Δtₜ, P, sys))
        append!(μ∂²s, μ∂zₜ∂zₜ₊₁fₜ(μₜ, zₜ, Δtₜ, P, sys))
    end

    return μ∂²s
end


function μ∂²F_structure(
    sys::QuantumSystem,
    T::Int,
    fourth_order_pade::Bool
)::Vector{Tuple{Int, Int}}

    structure = []

    for t = 1:T-1
        for rowcol in μ∂²zₜfₜstructure(sys, fourth_order_pade)
            rowcol_t = rowcol .+ index(t, 0, sys.vardim)
            push!(structure, rowcol_t)
        end

        for (row, col) in μ∂zₜ∂zₜ₊₁fₜstructure(sys)
            row_t = row + index(t, 0, sys.vardim)
            col_t = col + index(t + 1, 0, sys.vardim)
            push!(structure, (row_t, col_t))
        end
    end

    return structure
end


abstract type AbstractDynamics end

struct QuantumDynamics <: AbstractDynamics
    F::Function
    ∂F::Function
    ∂F_structure::Vector{Tuple{Int, Int}}
    μ∂²F::Union{Function, Nothing}
    μ∂²F_structure::Union{Vector{Tuple{Int, Int}}, Nothing}

    function QuantumDynamics(
        sys::QuantumSystem,
        integrator::Symbol,
        T::Int,
        Δt::Real;
        eval_hessian=true
    )
        P = eval(integrator)(sys)
        return new(
            Z -> F(Z, Δt, T, P, sys),
            Z -> ∂F(Z, Δt, T, P, sys),
            ∂F_structure(sys, T),
            eval_hessian ? (Z, μ) -> μ∂²F(Z, μ, Δt, T, P, sys) : nothing,
            eval_hessian ? μ∂²F_structure(sys, T, isa(P, FourthOrderPade)) : nothing
        )
    end


    function MinTimeQuantumDynamics(
        sys::AbstractSystem,
        integrator::Symbol,
        Z_indices::UnitRange{Int},
        Δt_indices::UnitRange{Int},
        T::Int;
        eval_hessian=true
    )
        P = eval(integrator)(sys)
        F̄ = Z̄ -> begin
            Δt = Z̄[Δt_indices]
            Z = Z̄[Z_indices]
            return F(Z, Δt, T, P, sys)
        end

        ∂F̄ = Z̄ -> begin
            Δt = Z̄[Δt_indices]
            Z = Z̄[Z_indices]
            return ∂F(Z, Δt, T, P, sys)
        end


    end


end



#
#
# min time sys dynamics (dynamical Δts)
#
#

function MinTimeQuantumDynamics(
    sys::AbstractSystem,
    integrator::Symbol,
    T::Int;
    eval_hessian=true
)

    P = eval(integrator)(sys)

    @views function f(
        zₜ::AbstractVector,
        zₜ₊₁::AbstractVector,
        Δtₜ::Real
    )
        xₜ₊₁ = zₜ₊₁[1:sys.nstates]
        xₜ = zₜ[1:sys.nstates]
        uₜ = zₜ[
            sys.n_wfn_states .+
            slice(sys.augdim + 1, sys.ncontrols)
        ]
        return dynamics(sys, P, xₜ₊₁, xₜ, uₜ, Δtₜ)
    end

    @views function F(Z::AbstractVector{F}) where F
        n_prob_variables = sys.vardim * T
        δX = zeros(F, sys.nstates * (T - 1))
        for t = 1:T-1
            zₜ = Z[slice(t, sys.vardim)]
            zₜ₊₁ = Z[slice(t + 1, sys.vardim)]
            Δtₜ = Z[n_prob_variables + t]
            δX[slice(t, sys.nstates)] = f(zₜ, zₜ₊₁, Δtₜ)
        end
        return δX
    end

    #
    # ∂zₜfₜ structure list
    #

    ∂zₜfₜ_structure = []


    # ∂ψ̃ⁱₜPⁱ blocks

    for i = 1:sys.nqstates
        for j = 1:sys.isodim     # jth column of ∂ψ̃ⁱₜPⁱ
            for k = 1:sys.isodim # kth row
                kj = (
                    index(i, k, sys.isodim),
                    index(i, j, sys.isodim)
                )
                push!(∂zₜfₜ_structure, kj)
            end
        end
    end


    # ∂aʲₜPⁱ blocks

    for i = 1:sys.nqstates
        for j = 1:sys.ncontrols  # jth column of ∂aʲₜPⁱ
            for k = 1:sys.isodim # kth row of ψ̃ⁱᵏₜ
                kj = (
                    index(i, k, sys.isodim),
                    sys.n_wfn_states + sys.∫a * sys.ncontrols + j
                )
                push!(∂zₜfₜ_structure, kj)
            end
        end
    end


    # -I blocks on main diagonal

    for k = 1:sys.n_aug_states
        kk = (k, k) .+ sys.n_wfn_states
        push!(∂zₜfₜ_structure, kk)
    end


    # -Δt⋅I blocks on shifted diagonal

    for k = 1:sys.n_aug_states
        kk_shifted = (k, k + sys.ncontrols) .+ sys.n_wfn_states
        push!(∂zₜfₜ_structure, kk_shifted)
    end



    #
    # Jacobian of f w.r.t. zₜ (see eq. 7)
    #

    ∂P = Jacobian(P)

    @views function ∂zₜfₜ(
        zₜ::AbstractVector,
        zₜ₊₁::AbstractVector,
        Δtₜ::Real
    )

        ∂s = []

        aₜ = zₜ[sys.n_wfn_states .+ slice(1 + sys.∫a, sys.ncontrols)]

        ∂ψ̃ⁱₜPⁱ = ∂P(aₜ, Δtₜ, false)

        for i = 1:sys.nqstates
            append!(∂s, ∂ψ̃ⁱₜPⁱ)
        end

        for i = 1:sys.nqstates

            ψ̃ⁱ_slice = slice(i, sys.isodim)

            for j = 1:sys.ncontrols

                ∂aʲₜPⁱ = ∂P(
                    zₜ₊₁[ψ̃ⁱ_slice],
                    zₜ[ψ̃ⁱ_slice],
                    aₜ,
                    Δtₜ,
                    j
                )

                append!(∂s, ∂aʲₜPⁱ)
            end
        end

        for _ = 1:sys.n_aug_states
            append!(∂s, -1.0)
        end

        for _ = 1:sys.n_aug_states
            append!(∂s, -Δtₜ)
        end

        return ∂s
    end



    #
    # ∂ₜ₊₁f structure list
    #

    ∂ₜ₊₁f_structure = []


    # ∂ψ̃ⁱₜPⁱ blocks

    for i = 1:sys.nqstates
        for j = 1:sys.isodim     # jth column: ∂ψ̃ⁱʲₜPⁱ
            for k = 1:sys.isodim # kth row: ∂ψ̃ⁱʲₜPⁱᵏ
                kj = (
                    index(i, k, sys.isodim),
                    index(i, j, sys.isodim)
                )
                push!(∂ₜ₊₁f_structure, kj)
            end
        end
    end


    # I for controls on main diagonal

    for k = 1:sys.n_aug_states
        kk = sys.n_wfn_states .+ (k, k)
        push!(∂ₜ₊₁f_structure, kk)
    end



    #
    # Jacobian of f w.r.t. zₜ₊₁ (see eq. 8)
    #

    @views function ∂ₜ₊₁f(zₜ::AbstractVector, Δtₜ::Real)

        ∂s = []

        aₜ = zₜ[sys.n_wfn_states .+ slice(1 + sys.∫a, sys.ncontrols)]

        ∂ψ̃ⁱₜ₊₁Pⁱ = ∂P(aₜ, Δtₜ, true)

        for i = 1:sys.nqstates
            append!(∂s, ∂ψ̃ⁱₜ₊₁Pⁱ)
        end

        for _ = 1:sys.n_aug_states
            append!(∂s, 1.0)
        end

        return ∂s
    end


    #
    # full system Jacobian
    #


    # ∂F structure list

    ∂F_structure = []

    for t = 1:T-1

        for (k, j) in ∂zₜfₜ_structure
            kₜ = k + index(t, 0, sys.nstates)
            jₜ = j + index(t, 0, sys.vardim)
            push!(∂F_structure, (kₜ, jₜ))
        end

        for (k, j) in ∂ₜ₊₁f_structure
            kₜ = k + index(t, 0, sys.nstates)
            jₜ = j + sys.vardim + index(t, 0, sys.vardim)
            push!(∂F_structure, (kₜ, jₜ))
        end
    end


    # full Jacobian

    @views function ∂F(Z::AbstractVector)
        ∂s = []
        for t = 1:T-1
            zₜ = Z[slice(t, sys.vardim)]
            zₜ₊₁ = Z[slice(t + 1, sys.vardim)]
            Δtₜ = Z[end - (T - 1) + t]
            append!(∂s, ∂zₜfₜ(zₜ, zₜ₊₁, Δtₜ))
            append!(∂s, ∂ₜ₊₁f(zₜ, Δtₜ))
        end
        return ∂s
    end


    #
    # dynamics Hessian
    #

    μ∂²F = nothing
    μ∂²F_structure = nothing

    if eval_hessian

        if P isa SecondOrderPade

            μ∂²F_structure = []


            for t = 1:T-1
                for i = 1:sys.nqstates

                    # ∂ψ̃ⁱₜ∂aʲₜPⁱ block:

                    for j = 1:sys.ncontrols
                        for k = 1:sys.isodim
                            kⁱjₜ = (
                                # kⁱth row
                                index(t, 0, sys.vardim) +
                                index(i, k, sys.isodim),

                                # jth column
                                index(t, 0, sys.vardim) +
                                sys.n_wfn_states +
                                index(1 + sys.∫a, j, sys.ncontrols)
                            )
                            push!(μ∂²F_structure, kⁱjₜ)
                        end
                    end


                    # ∂aᵏₜ∂ψ̃ⁱₜ₊₁Pⁱ block

                    for k = 1:sys.ncontrols
                        for j = 1:sys.isodim
                            jⁱkₜ₊₁ = (
                                # kth row
                                index(t, 0, sys.vardim) +
                                sys.n_wfn_states +
                                index(1 + sys.∫a, k, sys.ncontrols),

                                # jⁱth column
                                index(t + 1, 0, sys.vardim) +
                                index(i, j, sys.isodim)
                            )
                            push!(μ∂²F_structure, jⁱkₜ₊₁)
                        end
                    end
                end
            end


            H = SecondOrderPadeHessian(sys)

            μ∂²F = @views (
                Z::AbstractVector,
                μ::AbstractVector
            ) -> begin
                Hⁱᵏʲs = []

                for t = 1:T-1
                    μₜ = μ[slice(t, sys.nstates)]
                    Δtₜ = Z[end - (T - 1) + t]
                    for i = 1:sys.nqstates

                        # ∂ψ̃ⁱₜ∂aʲₜPⁱ block
                        for j = 1:sys.ncontrols
                            append!(
                                Hⁱᵏʲs,
                                H(μₜ, Δtₜ, i, j, false)
                            )
                        end

                        # ∂ψ̃ⁱₜ₊₁∂aʲₜPⁱ block
                        for k = 1:sys.ncontrols
                            append!(
                                Hⁱᵏʲs,
                                H(μₜ, Δtₜ, i, k, true)
                            )
                        end
                    end
                end
                return Hⁱᵏʲs
            end

        elseif P isa FourthOrderPade

            μ∂²F_structure = []

            for t = 1:T-1

                # ∂aᵏₜ∂aʲₜPⁱ block

                for j = 1:sys.ncontrols
                    for k = 1:j
                        kjₜ = (
                            index(t, 0, sys.vardim) +
                            sys.n_wfn_states +
                            index(1 + sys.∫a, k, sys.ncontrols),

                            index(t, 0, sys.vardim) +
                            sys.n_wfn_states +
                            index(1 + sys.∫a, k, sys.ncontrols)
                        )
                        push!(μ∂²F_structure, kjₜ)
                    end
                end


                # ∂ψ̃ⁱₜ₍₊₁₎∂aʲₜPⁱ blocks

                for i = 1:sys.nqstates

                    # ∂ψ̃ⁱₜ∂aʲₜPⁱ block:

                    for j = 1:sys.ncontrols
                        for k = 1:sys.isodim
                            kⁱjₜ = (
                                # kⁱth row
                                index(t, 0, sys.vardim) +
                                index(i, k, sys.isodim),

                                # jth column
                                index(t, 0, sys.vardim) +
                                sys.n_wfn_states +
                                index(1 + sys.∫a, j, sys.ncontrols)
                            )
                            push!(μ∂²F_structure, kⁱjₜ)
                        end
                    end


                    # ∂aᵏₜ∂ψ̃ⁱₜ₊₁Pⁱ block

                    for k = 1:sys.ncontrols
                        for j = 1:sys.isodim
                            jⁱkₜ₊₁ = (
                                # kth row
                                index(t, 0, sys.vardim) +
                                sys.n_wfn_states +
                                index(1 + sys.∫a, k, sys.ncontrols),

                                # jⁱth column
                                index(t + 1, 0, sys.vardim) +
                                index(i, j, sys.isodim)
                            )
                            push!(μ∂²F_structure, jⁱkₜ₊₁)
                        end
                    end
                end
            end

            H = FourthOrderPadeHessian(sys)

            μ∂²F = @views (
                Z::AbstractVector,
                μ::AbstractVector
            ) -> begin
                Hs = []
                for t = 1:T-1

                    ψ̃ₜ₊₁ = Z[
                        slice(
                            t + 1,
                            sys.n_wfn_states,
                            sys.vardim
                        )
                    ]

                    ψ̃ₜ = Z[
                        slice(
                            t,
                            sys.n_wfn_states,
                            sys.vardim
                        )
                    ]

                    # dim(μ) = sys.nstates * (T - 1)
                    μₜ = μ[slice(t, sys.nstates)]

                    Δtₜ = Z[end - (T - 1) + t]


                    # ∂aᵏₜ∂aʲₜPⁱ block

                    for j = 1:sys.ncontrols
                        for k = 1:j
                            Hᵏʲ = H(ψ̃ₜ₊₁, ψ̃ₜ, μₜ, Δtₜ, k, j)
                            append!(Hs, Hᵏʲ)
                        end
                    end

                    aₜ = Z[
                        index(
                            t,
                            sys.n_wfn_states,
                            sys.vardim
                        ) .+

                        slice(1 + sys.∫a, sys.ncontrols)
                    ]

                    # ∂ψ̃ⁱₜ₍₊₁₎∂aʲₜPⁱ blocks

                    for i = 1:sys.nqstates

                        # ∂ψ̃ⁱₜ∂aʲₜPⁱ block
                        for j = 1:sys.ncontrols
                            append!(
                                Hs,
                                H(aₜ, μₜ, Δtₜ, i, j, false)
                            )
                        end

                        # ∂ψ̃ⁱₜ₊₁∂aʲₜPⁱ block
                        for k = 1:sys.ncontrols
                            append!(
                                Hs,
                                H(aₜ, μₜ, Δtₜ, i, k, true)
                            )
                        end
                    end
                end
                return Hs
            end
        end
    end

    return QuantumDynamics(
        F,
        ∂F,
        ∂F_structure,
        μ∂²F,
        μ∂²F_structure
    )
end


end
