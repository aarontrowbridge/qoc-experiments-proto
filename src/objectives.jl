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
    ∇²L::Function
    ∇²L_structure::Vector{Tuple{Int,Int}}
end

function SystemObjective(
    system::AbstractQubitSystem,
    loss::Function,
    T::Int,
    Q::Float64,
    Qf::Float64,
    R::Float64,
    eval_hessian::Bool
)

    ls = [ψ̃ -> loss(ψ̃, system.ψ̃f[slice(i, system.isodim)]) for i = 1:system.nqstates]

    @views function system_loss(ψ̃s)
        sum([l(ψ̃s[slice(i, system.isodim)]) for (i, l) in enumerate(ls)])
    end

    # TODO: add quadratic loss terms for augmented state vars

    @views function L(Z)
        ψ̃ts = [Z[slice(t, system.n_wfn_states, system.vardim)] for t = 1:T]
        us = zeros(system.ncontrols * T)
        for t = 1:T
            us[slice(t, system.ncontrols)] =
                Z[slice(t, system.nstates + 1, system.vardim, system.vardim)]
        end
        Qs = [fill(Q, T-1); Qf]
        return dot(Qs, system_loss.(ψ̃ts)) + R / 2 * dot(us, us)
    end

    Symbolics.@variables ψ[1:system.isodim]
    ψ = collect(ψ)

    ∇ls_symb = [Symbolics.gradient(l(ψ), ψ) for l in ls]
    ∇ls_expr = [Symbolics.build_function(∇l, ψ) for ∇l in ∇ls_symb]
    ∇ls = [eval(∇l_expr[1]) for ∇l_expr in ∇ls_expr]

    @views function ∇L(Z)
        ∇ = zeros(system.vardim*T)
        zs = [Z[slice(t, system.vardim)] for t in 1:T]
        ψ̃s = [z[1:system.n_wfn_states] for z in zs]
        us = [z[end - system.ncontrols + 1:end] for z in zs]
        Qs = [fill(Q, T-1); Qf]
        for t = 1:T
            for j = 1:system.nqstates
                ψⱼinds = slice(j, system.isodim)
                ∇[index(t, 0, system.vardim) .+ ψⱼinds] = Qs[t] * ∇ls[j](ψ̃s[t][ψⱼinds])
            end
            for k = 1:system.ncontrols
                ∇[index(t, system.nstates + k, system.vardim)] = R * us[t][k]
            end
        end
        return ∇
    end

    # TODO: fix Hessian
    if eval_hessian
        ∇²L_symb = Symbolics.sparsehessian(L(y), y)
        I, J, _ = findnz(∇²L_symb)
        ∇²L_structure = collect(zip(I, J))
        ∇²L_expr = Symbolics.build_function(∇²L_symb, y)
        ∇²L = eval(∇²L_expr[1])
    else
        ∇²L = (_) -> nothing
        ∇²L_structure = []
    end

    return SystemObjective(L, ∇L, ∇²L, ∇²L_structure)
end


# TODO: implement Hessian for MinTimeObjective

struct MinTimeObjective
    L::Function
    ∇L::Function
end

function MinTimeObjective(T::Int, vardim::Int, Rᵤ::Float64, Rₛ::Float64)

    total_time(z) = sum([z[index(t, vardim + 1)] for t = 1:T-1])

    # TODO: chase down variables and remove end control variable

    function u_smoothness_regulator(z)
        ∑Δu² = 0.0
        for t = 1:T-2
            uₜ₊₁ = z[index(t + 1, vardim, vardim + 1)]
            uₜ = z[index(t, vardim, vardim + 1)]
            Δu = uₜ₊₁ - uₜ
            ∑Δu² += Δu^2
        end
        return 0.5 * Rₛ * ∑Δu²
    end

    function u_amplitude_regulator(z)
        ∑u² = 0.0
        for t = 1:T-1
            uₜ = z[index(t, vardim, vardim + 1)]
            ∑u² += uₜ^2
        end
        return 0.5 * Rᵤ * ∑u²
    end

    L = z -> total_time(z) + u_smoothness_regulator(z) + u_amplitude_regulator(z)

    function ∇L(z)
        ∇ = zeros((vardim + 1) * (T - 1) + vardim)

        for t = 1:T-1
            ∇[index(t, vardim + 1)] = 1.0
            ∇[index(t, vardim, vardim + 1)] = Rᵤ * z[index(t, vardim, vardim + 1)]
        end

        for t = 1:T-2
            uₜ₊₁ = z[index(t + 1, vardim, vardim + 1)]
            uₜ = z[index(t, vardim, vardim + 1)]
            Δu = uₜ₊₁ - uₜ
            ∇[index(t, vardim, vardim + 1)] += -Rₛ * Δu
            ∇[index(t + 1, vardim, vardim + 1)] += Rₛ * Δu
        end

        return ∇
    end

    return MinTimeObjective(L, ∇L)
end




end
