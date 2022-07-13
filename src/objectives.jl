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
    system::AbstractQubitSystem{N},
    loss::Function,
    T::Int,
    Q::Float64,
    Qf::Float64,
    R::Float64,
    eval_hessian::Bool
) where N

    vardim = system.nstates + 1

    ls = [ψ̃ -> loss(ψ̃, system.ψ̃goal[slice(i, system.isodim)]) for i = 1:system.nqstates]

    @views function system_loss(ψ̃s)
        sum([l(ψ̃s[slice(i, system.isodim)]) for (i, l) in enumerate(ls)])
    end

    @views function L(Z)
        aug = system.control_order + 2
        ψ̃ts = [Z[slice(t, vardim; stretch=-aug)] for t in 1:T]
        us = [Z[index(t, vardim)] for t in 1:T]
        Qs = [fill(Q, T-1); Qf]
        return dot(Qs, system_loss.(ψ̃ts)) + R / 2 * dot(us, us)
    end

    Symbolics.@variables ψ[1:system.isodim]
    ψ = collect(ψ)

    ∇ls_symb = [Symbolics.gradient(l(ψ), ψ) for l in ls]
    ∇ls_expr = [Symbolics.build_function(∇l, ψ) for ∇l in ∇ls_symb]
    ∇ls = [eval(∇l_expr[1]) for ∇l_expr in ∇ls_expr]

    @views function ∇L(Z)
        ∇ = zeros(vardim*T)
        zs = [Z[slice(t, vardim)] for t in 1:T]
        ψ̃s = [z[1:system.isodim*system.nqstates] for z in zs]
        us = [z[end] for z in zs]
        Qs = [fill(Q, T-1); Qf]
        for t = 1:T
            for k = 1:system.nqstates
                ψₖinds = slice(k, system.isodim)
                ∇[ψₖinds .+ vardim * (t - 1)] = Qs[t] * ∇ls[k](ψ̃s[t][ψₖinds])
            end
            ∇[t*vardim] = R * us[t]
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
