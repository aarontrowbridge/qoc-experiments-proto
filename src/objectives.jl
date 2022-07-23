module Objectives

export SystemObjective
export MinTimeObjective

using ..Utils
using ..QubitSystems
using ..Losses

using LinearAlgebra
using SparseArrays
using Symbolics

#
# objective functions
#

struct SystemObjective
    L::Function
    ∇L::Function
    ∇²L::Union{Function, Nothing}
    ∇²L_structure::Union{Vector{Tuple{Int,Int}}, Nothing}
end

function SystemObjective(
    system::AbstractQubitSystem,
    loss::QuantumStateLoss,
    T::Int,
    Q::Float64,
    R::Float64,
    eval_hessian::Bool
)

    @views function L(Z)
        ψ̃T = Z[slice(T, system.n_wfn_states, system.vardim)]
        us = zeros(system.ncontrols * T)
        for t = 1:T
            us[slice(t, system.ncontrols)] =
                Z[slice(t, system.nstates + 1, system.vardim, system.vardim)]
        end
        return Q * loss(ψ̃T) + R / 2 * dot(us, us)
    end

    ∇l = QuantumStateLossGradient(loss)

    # this version of ∇L removes intermediate Qs from objective
    @views function ∇L(Z)
        ∇ = zeros(system.vardim * T)

        for t = 1:T-1
            uₜ_slice = slice(t, system.nstates + 1, system.vardim, system.vardim)
            uₜ = Z[uₜ_slice]
            ∇[uₜ_slice] = R * uₜ
        end

        ψ̃T_slice = slice(T, system.n_wfn_states, system.vardim)
        ψ̃T = Z[ψ̃T_slice]
        ∇[ψ̃T_slice] = Q * ∇l(ψ̃T)

        return ∇
    end

    if eval_hessian
        ∇²l = QuantumStateLossHessian(loss)

        ∇²L_structure = []

        # uₜ Hessian structure (eq. 17)
        for t = 1:T-1
            offset = index(t, system.nstates, system.vardim)
            KK = (1:system.ncontrols) .+ offset
            append!(∇²L_structure, collect(zip(KK, KK)))
        end

        # ℓⁱs Hessian structure (eq. 17)
        append!(∇²L_structure, structure(∇²l, T, system.vardim))

        function ∇²L(Z::AbstractVector)
            Hs = fill(R, system.ncontrols * (T - 1))
            ψ̃T = view(Z, slice(T, system.n_wfn_states, system.vardim))
            append!(Hs, ∇²l(ψ̃T))
            return Hs
        end
    else
        ∇²L = nothing
        ∇²L_structure = nothing
    end

    return SystemObjective(L, ∇L, ∇²L, ∇²L_structure)
end


# TODO: implement Hessian for MinTimeObjective

struct MinTimeObjective
    L::Function
    ∇L::Function
end

function MinTimeObjective(sys::AbstractQubitSystem, T::Int, Rᵤ::Float64, Rₛ::Float64)

    total_time(z) = sum([z[index(t, sys.vardim + 1)] for t = 1:T-1])

    # TODO: chase down variables and remove end control variable

    @views function u_smoothness_regulator(z)
        ∑Δu² = 0.0
        for t = 1:T-2
            uₜ₊₁ = z[slice(t + 1, sys.nstates + 1, sys.vardim, sys.vardim + 1)]
            uₜ = z[slice(t, sys.nstates + 1, sys.vardim, sys.vardim + 1)]
            Δu = uₜ₊₁ - uₜ
            ∑Δu² += dot(Δu, Δu)
        end
        return 0.5 * Rₛ * ∑Δu²
    end

    @views function u_amplitude_regulator(z)
        ∑u² = 0.0
        for t = 1:T-1
            uₜ = z[slice(t, sys.nstates + 1, sys.vardim, sys.vardim + 1)]
            ∑u² += dot(uₜ, uₜ)
        end
        return 0.5 * Rᵤ * ∑u²
    end

    L = z -> total_time(z) + u_smoothness_regulator(z) + u_amplitude_regulator(z)

    @views function ∇L(z)

        ∇ = zeros((sys.vardim + 1) * (T - 1) + sys.vardim)

        for t = 1:T-1
            ∇[index(t, sys.vardim + 1)] = 1.0
            ∇[slice(t, sys.nstates + 1, sys.vardim, sys.vardim + 1)] =
                Rᵤ * z[slice(t, sys.nstates + 1, sys.vardim, sys.vardim + 1)]
        end

        for t = 1:T-2
            uₜ₊₁ = z[slice(t + 1, sys.nstates + 1, sys.vardim, sys.vardim + 1)]
            uₜ = z[slice(t, sys.nstates + 1, sys.vardim, sys.vardim + 1)]
            Δu = uₜ₊₁ - uₜ
            ∇[slice(t, sys.nstates + 1, sys.vardim, sys.vardim + 1)] += -Rₛ * Δu
            ∇[slice(t + 1, sys.nstates + 1, sys.vardim, sys.vardim + 1)] += Rₛ * Δu
        end

        return ∇
    end

    return MinTimeObjective(L, ∇L)
end




end
