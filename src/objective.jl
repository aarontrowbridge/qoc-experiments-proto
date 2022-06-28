module Objective

export objective

export SystemObjective

using ..Utils
using ..QubitSystems

using LinearAlgebra
using SparseArrays
using Symbolics

#
# objective functions
#

@views function objective(system::AbstractQubitSystem{N}, loss, xs, us, T, Q, Qf, R) where N
    ψ̃ts = [xs[t][1:(system.isodim*system.nqstates)] for t = 1:T]
    u = vcat(us...)
    Qs = [fill(Q, T-1); Qf]
    return dot(Qs, loss.(ψ̃ts)) + R / 2 * dot(u, u)
end

struct SystemObjective
    L::Function
    ∇L::Function
    ∇²L::Function
    ∇²L_structure::Vector{Tuple{Int,Int}}
end

function SystemObjective(
    system::AbstractQubitSystem{N},
    loss::Function,
    eval_hessian::Bool,
    T::Int,
    Q::Float64,
    Qf::Float64,
    R::Float64
) where N

    vardim = system.nstates + 1

    ls = [ψ̃ -> loss(ψ̃, system.ψ̃goal[slice(i, system.isodim)]) for i = 1:system.nqstates]

    @views function system_loss(ψ̃s)
        sum([l(ψ̃s[slice(i, system.isodim)]) for (i, l) in enumerate(ls)])
    end

    @views function L(z)
        xus = [z[slice(t, vardim)] for t in 1:T]
        xs = [xu[1:end-1] for xu in xus]
        us = [xu[end:end] for xu in xus]
        return objective(system, system_loss, xs, us, T, Q, Qf, R)
    end

    Symbolics.@variables ψ[1:system.isodim]
    ψ = collect(ψ)

    ∇ls_symb = [Symbolics.gradient(l(ψ), ψ) for l in ls]
    ∇ls_expr = [Symbolics.build_function(∇l, ψ) for ∇l in ∇ls_symb]
    ∇ls = [eval(∇l_expr[1]) for ∇l_expr in ∇ls_expr]

    function ∇L(z)
        ∇ = zeros(vardim*T)
        xus = [z[slice(t, vardim)] for t in 1:T]
        ψ̃s = [xu[1:system.isodim*system.nqstates] for xu in xus]
        us = [xu[end] for xu in xus]
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


    # Symbolics.@variables y[1:(system.nstates+1)*T]

    # y = collect(y)

    # ∇L_symb = Symbolics.gradient(L(y), y)
    # ∇L_expr = Symbolics.build_function(∇L_symb, y)
    # ∇L = eval(∇L_expr[1])

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

end
