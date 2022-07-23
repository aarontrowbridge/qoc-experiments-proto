module Objectives

export SystemObjective
export MinTimeObjective

using ..Utils
using ..QubitSystems

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
    loss::Function,
    T::Int,
    Q::Float64,
    R::Float64,
    eval_hessian::Bool
)

    # ls = [ψ̃ -> loss(ψ̃, system.ψ̃goal[slice(i, system.isodim)]) for i = 1:system.nqstates]

    # @views function system_loss(ψ̃s)
    #     sum([l(ψ̃s[slice(i, system.isodim)]) for (i, l) in enumerate(ls)])
    # end

    # TODO: add quadratic loss terms for augmented state vars

    # @views function L(Z)
    #     ψ̃ts = [Z[slice(t, system.n_wfn_states, system.vardim)] for t = 1:T]
    #     us = zeros(system.ncontrols * T)
    #     for t = 1:T
    #         us[slice(t, system.ncontrols)] =
    #             Z[slice(t, system.nstates + 1, system.vardim, system.vardim)]
    #     end
    #     Qs = [fill(Q, T-1); Qf]
    #     return dot(Qs, system_loss.(ψ̃ts)) + R / 2 * dot(us, us)
    # end


    system_loss = QuantumStateLoss(system; loss=loss)

    @views function L(Z)
        ψ̃T = Z[slice(T, system.n_wfn_states, system.vardim)]
        us = zeros(system.ncontrols * T)
        for t = 1:T
            us[slice(t, system.ncontrols)] =
                Z[slice(t, system.nstates + 1, system.vardim, system.vardim)]
        end
        return Q * system_loss(ψ̃T) + R / 2 * dot(us, us)
    end

    ∇l = QuantumStateLossGradient(system_loss)

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
        ∇²l = QuantumStateLossHessian(system_loss)

        ∇²L_structure = []

        # aₜ Hessian structure (eq. 17)
        for t = 1:T-1
            offset = index(t, system.n_wfn_states, system.vardim)
            idxs = slice(2, system.ncontrols) .+ offset
            append!(∇²L_structure, collect(zip(idxs, idxs)))
        end

        # ℓⁱ Hessian structure (eq. 17)
        append!(∇²L_structure, structure(∇²l, T))

        function ∇²L(Z::Vector{Real})
            Hs = fill(R, system.ncontrols * (T - 1))
            ψ̃T = view(Z, slice(T, system.n_wfn_states, system.vardim))
            append!(Hs, ∇²l(ψ̃T))
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
