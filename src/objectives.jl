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
        us = [Z[t*vardim] for t in 1:T]
        Qs = [fill(Q, T-1); Qf]
        return dot(Qs, system_loss.(ψ̃ts)) + R / 2 * dot(us, us)
    end

    Symbolics.@variables ψ[1:system.isodim]
    ψ = collect(ψ)

    ∇ls_symb = [Symbolics.gradient(l(ψ), ψ) for l in ls]
    ∇ls_expr = [Symbolics.build_function(∇l, ψ) for ∇l in ∇ls_symb]
    ∇ls = [eval(∇l_expr[1]) for ∇l_expr in ∇ls_expr]

    function ∇L(Z)
        ∇ = zeros(vardim*T)
        zs = [Z[slice(t, vardim)] for t in 1:T]
        ψ̃s = [z[1:system.isodim*system.nqstates] for z in zs]
        us = [z[end] for z in zs]
        Qs = [fill(Q, T-1); Qf]
        for t = 1:T
            for k = 1:system.nqstates
                ψₖinds = slice(k, system.isodim)
                ∇[ψₖinds .+ (t-1) * vardim] = Qs[t] * ∇ls[k](ψ̃s[t][ψₖinds])
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

function MinTimeObjective(T::Int, vardim::Int, B::Float64, squared_loss::Bool)
    total_time(z) = sum([z[t*(vardim + 1)] for t = 1:T-1])

    if squared_loss
        L = z -> 0.5 * B * total_time(z)^2
        ∇L = z -> vcat(
            [[zeros(vardim); B * total_time(z)] for _ = 1:T-1]...,
            zeros(vardim)
        )
    else
        L = z -> B * total_time(z)
        ∇L = z -> vcat(
            [[zeros(vardim); B] for _ = 1:T-1]...,
            zeros(vardim)
        )
    end

    return MinTimeObjective(L, ∇L)
end




end
