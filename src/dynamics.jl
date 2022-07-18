module Dynamics

export SystemDynamics
export MinTimeSystemDynamics

using ..Utils
using ..QuantumLogic
using ..QubitSystems

using LinearAlgebra
using SparseArrays
using Symbolics

#
# dynamics functions
#

@views function dynamics(
    sys::AbstractQubitSystem,
    integrator::Function,
    xₜ₊₁,
    xₜ,
    uₜ,
    Δt
)
    augₜs = xₜ[(end - sys.n_aug_states + 1):end]
    augₜ₊₁s = xₜ₊₁[(end - sys.n_aug_states + 1):end]

    âugₜ₊₁s = zeros(typeof(xₜ[1]), sys.n_aug_states)

    for i = 1:sys.augdim
        for k = 1:sys.ncontrols
            idxᵢₖ = index(i, k, sys.ncontrols)
            if i != sys.augdim
                idxᵢ₊₁ₖ = index(i + 1, k, sys.ncontrols)
                âugₜ₊₁s[idxᵢₖ] = augₜs[idxᵢₖ] + Δt * augₜs[idxᵢ₊₁ₖ]
            else
                âugₜ₊₁s[idxᵢₖ] = augₜs[idxᵢₖ] + Δt * uₜ[k]
            end
        end
    end

    δaₜ₊₁s = augₜ₊₁s - âugₜ₊₁s

    δψ̃ₜ₊₁s = zeros(typeof(xₜ[1]), sys.n_wfn_states)

    aₜs = xₜ[sys.n_wfn_states .+ slice(2, sys.ncontrols)]

    for i = 1:sys.nqstates
        ψ̃ᵢslice = slice(i, sys.isodim)
        δψ̃ₜ₊₁s[ψ̃ᵢslice] = integrator(xₜ₊₁[ψ̃ᵢslice], xₜ[ψ̃ᵢslice], aₜs, Δt)
    end

    δxₜ₊₁ = vcat(δψ̃ₜ₊₁s..., δaₜ₊₁s)

    return δxₜ₊₁
end

struct SystemDynamics
    f::Function
    ∇f::Function
    ∇f_structure::Vector{Tuple{Int, Int}}
    ∇²f::Function
    ∇²f_structure::Vector{Tuple{Int, Int}}
end

function SystemDynamics(
    sys::AbstractQubitSystem,
    integrator::Function,
    T::Int,
    Δt::Float64,
    eval_hessian::Bool
)

    function sys_integrator(ψ̃ₜ₊₁, ψ̃ₜ, aₜs, Δt)
        return integrator(ψ̃ₜ₊₁, ψ̃ₜ, aₜs, Δt, sys.G_drift, sys.G_drives)
    end

    @views function fₜ(zₜ₊₁, zₜ)
        xₜ₊₁ = zₜ₊₁[1:sys.nstates]
        xₜ = zₜ[1:sys.nstates]
        uₜs = zₜ[end - sys.ncontrols + 1:end]
        return dynamics(sys, sys_integrator, xₜ₊₁, xₜ, uₜs, Δt)
    end

    @views function f(Z)
        δxs = zeros(typeof(Z[1]), sys.nstates * (T - 1))
        for t = 1:T-1
            δxₜ₊₁ = fₜ(Z[slice(t + 1, sys.vardim)], Z[slice(t, sys.vardim)])
            δxs[slice(t, sys.nstates)] = δxₜ₊₁
        end
        return δxs
    end

    Symbolics.@variables zz[1:(sys.vardim * 2)]

    zz = collect(zz)

    f̂ₜ(zₜzₜ₊₁) = fₜ(zₜzₜ₊₁[sys.vardim+1:end], zₜzₜ₊₁[1:sys.vardim])

    ∇f̂ₜ_symb = Symbolics.sparsejacobian(f̂ₜ(zz), zz)

    Is = findnz(∇f̂ₜ_symb)[1]
    Js = findnz(∇f̂ₜ_symb)[2]

    ∇f_structure = []
    for t = 1:T-1
        idxs = zip(
            Is .+ index(t, 0, sys.nstates),
            Js .+ index(t, 0, sys.vardim)
        )
        append!(∇f_structure, collect(idxs))
    end

    ∇f̂ₜ_expr = Symbolics.build_function(∇f̂ₜ_symb, zz)
    ∇f̂ₜ = eval(∇f̂ₜ_expr[1])

    function ∇f(Z)
        jac = spzeros(sys.nstates*(T-1), sys.vardim*T)
        for t = 1:T-1
            var_idxs = slice(t, sys.vardim; stretch=sys.vardim)
            jac[slice(t, sys.nstates), var_idxs] = ∇f̂ₜ(Z[var_idxs])
        end
        return jac
    end

    # TODO: fix this
    if eval_hessian
        Symbolics.@variables μ[1:sys.nstates * (T - 1)]
        μ = collect(μ)
        ∇²f_symb = Symbolics.sparsehessian(dot(μ, f(y)), [y; μ])
        I, J, _ = findnz(∇²f_symb)
        ∇²f_structure = collect(zip(I, J))
        ∇²f_expr = Symbolics.build_function(∇²f_symb, [y; μ])
        ∇²f_eval = eval(∇²f_expr[1])
        ∇²f = (ŷ, μ̂) -> ∇²f_eval([ŷ; μ̂])
    else
        ∇²f = (_, _) -> nothing
        ∇²f_structure = []
    end

    return SystemDynamics(f, ∇f, ∇f_structure, ∇²f, ∇²f_structure)
end

# min time sys dynamics (dynamical Δts)
function MinTimeSystemDynamics(
    sys::AbstractQubitSystem,
    integrator::Function,
    T::Int,
    eval_hessian::Bool
)

    vardim = sys.nstates + 1

    function sys_integrator(ψ̃ₜ₊₁, ψ̃ₜ, aₜ, Δt)
        return integrator(ψ̃ₜ₊₁, ψ̃ₜ, aₜ, Δt, sys.G_drift, sys.G_drive)
    end

    @views function fₜ(xuₜ₊₁, xuₜ, Δt)
        xₜ₊₁ = xuₜ₊₁[1:end-1]
        xₜ = xuₜ[1:end-1]
        uₜ = xuₜ[end:end]
        return dynamics(sys, sys_integrator, xₜ₊₁, xₜ, uₜ, Δt)
    end

    @views function f(z)
        zs = [z[slice(t, vardim + 1; stretch=-1)] for t in 1:T]
        Δts = [z[index(t, vardim + 1)] for t in 1:T-1]
        δxs = zeros(typeof(z[1]), sys.nstates * (T - 1))
        for t = 1:T-1
            δxₜ₊₁ = fₜ(zs[t+1], zs[t], Δts[t])
            δxs[slice(t, sys.nstates)] = δxₜ₊₁
        end
        return δxs
    end

    Symbolics.@variables zΔtz[1:(2*vardim + 1)]

    zΔtz = collect(zΔtz)

    @views f̂ₜ(zₜΔtzₜ₊₁) = fₜ(
        zₜΔtzₜ₊₁[vardim+2:end], # zₜ₊₁
        zₜΔtzₜ₊₁[1:vardim],     # zₜ
        zₜΔtzₜ₊₁[vardim+1],     # Δt
    )

    ∇f̂ₜ_symb = Symbolics.sparsejacobian(f̂ₜ(zΔtz), zΔtz)

    Is = findnz(∇f̂ₜ_symb)[1]
    Js = findnz(∇f̂ₜ_symb)[2]

    ∇f_structure = vcat(
        [
            collect(
                zip(
                    Is .+ (t - 1) * sys.nstates,
                    Js .+ (t - 1) * (vardim + 1)
                )
            ) for t = 1:T-1
        ]...
    )

    ∇f̂ₜ_expr = Symbolics.build_function(∇f̂ₜ_symb, zΔtz)
    ∇f̂ₜ = eval(∇f̂ₜ_expr[1])

    function ∇f(Z)
        jac = spzeros(sys.nstates*(T - 1), (vardim + 1)*T)
        for t = 1:T-1
            zₜΔtzₜ₊₁_inds = slice(t, vardim+1; stretch=vardim)
            zₜΔtzₜ₊₁ = Z[zₜΔtzₜ₊₁_inds]
            ∇fₜ = ∇f̂ₜ(zₜΔtzₜ₊₁)
            jac[slice(t, sys.nstates), zₜΔtzₜ₊₁_inds] = ∇fₜ
        end
        return jac
    end

    # TODO: redo hessian
    if eval_hessian
        Symbolics.@variables μ[1:sys.nstates * (T - 1)]
        μ = collect(μ)
        ∇²f_symb = Symbolics.sparsehessian(dot(μ, f(y)), [y; μ])
        I, J, _ = findnz(∇²f_symb)
        ∇²f_structure = collect(zip(I, J))
        ∇²f_expr = Symbolics.build_function(∇²f_symb, [y; μ])
        ∇²f_eval = eval(∇²f_expr[1])
        ∇²f = (ŷ, μ̂) -> ∇²f_eval([ŷ; μ̂])
    else
        ∇²f = (_, _) -> nothing
        ∇²f_structure = []
    end

    return SystemDynamics(f, ∇f, ∇f_structure, ∇²f, ∇²f_structure)
end


end
