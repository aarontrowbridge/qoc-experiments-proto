module Dynamics

export dynamics
export SystemDynamics

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
    system::AbstractQubitSystem{N},
    integrator,
    xₜ₊₁,
    xₜ,
    uₜ,
    Δt
) where N
    aₜs = xₜ[(end-system.control_order):end]
    aₜ₊₁s = xₜ₊₁[(end-system.control_order):end]

    âₜ₊₁s = zeros(typeof(xₜ[1]), system.control_order + 1)

    for i = 1:system.control_order
        âₜ₊₁s[i] = aₜs[i] + Δt * aₜs[i + 1]
    end

    âₜ₊₁s[end] = aₜs[end] + Δt * uₜ[1]

    δaₜ₊₁s = aₜ₊₁s - âₜ₊₁s

    aₜ = xₜ[end-system.control_order+1]
    δψ̃ₜ₊₁s = [
        integrator(
            xₜ₊₁[slice(i, system.isodim)],
            xₜ[slice(i, system.isodim)],
            aₜ,
            Δt
        ) for i = 1:system.nqstates
    ]

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
    system::AbstractQubitSystem{N},
    integrator::Function,
    T::Int,
    Δt::Float64,
    eval_hessian::Bool
) where N

    vardim = system.nstates + 1

    function system_integrator(ψ̃ₜ₊₁, ψ̃ₜ, aₜ, Δt)
        return integrator(ψ̃ₜ₊₁, ψ̃ₜ, aₜ, Δt, system.G_drift, system.G_drive)
    end

    @views function fₜ(zₜ₊₁, zₜ)
        xₜ₊₁ = zₜ₊₁[1:end-1]
        xₜ = zₜ[1:end-1]
        uₜ = zₜ[end:end]
        return dynamics(system, system_integrator, xₜ₊₁, xₜ, uₜ, Δt)
    end

    @views function f(Z)
        δxs = zeros(typeof(Z[1]), system.nstates * (T - 1))
        for t = 1:T-1
            δxₜ₊₁ = fₜ(Z[slice(t+1, vardim)], Z[slice(t, vardim)])
            δxs[slice(t, system.nstates)] = δxₜ₊₁
        end
        return δxs
    end

    Symbolics.@variables zz[1:2*vardim]

    zz = collect(zz)

    f̂ₜ(zₜzₜ₊₁) = fₜ(zₜzₜ₊₁[vardim+1:end], zₜzₜ₊₁[1:vardim])

    ∇f̂ₜ_symb = Symbolics.sparsejacobian(f̂ₜ(zz), zz)

    Is = findnz(∇f̂ₜ_symb)[1]
    Js = findnz(∇f̂ₜ_symb)[2]

    ∇f_structure = vcat(
        [
            collect(
                zip(
                    Is .+ (t - 1) * system.nstates,
                    Js .+ (t - 1) * vardim
                )
            ) for t = 1:T-1
        ]...
    )

    ∇f̂ₜ_expr = Symbolics.build_function(∇f̂ₜ_symb, zz)
    ∇f̂ₜ = eval(∇f̂ₜ_expr[1])

    function ∇f(Z)
        jac = spzeros(system.nstates*(T-1), vardim*T)
        for t = 1:T-1
            ∇fₜ = ∇f̂ₜ(Z[slice(t, vardim; stretch=vardim)])
            jac[slice(t, system.nstates), slice(t, vardim; stretch=vardim)] = ∇fₜ
        end
        return jac
    end

    # TODO: fix this
    if eval_hessian
        Symbolics.@variables μ[1:system.nstates * (T - 1)]
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

# min time system dynamics (dynamical Δts)
function SystemDynamics(
    system::AbstractQubitSystem{N},
    integrator::Function,
    T::Int,
    eval_hessian::Bool
) where N

    vardim = system.nstates + 1

    function system_integrator(ψ̃ₜ₊₁, ψ̃ₜ, aₜ, Δt)
        return integrator(ψ̃ₜ₊₁, ψ̃ₜ, aₜ, Δt, system.G_drift, system.G_drive)
    end

    @views function fₜ(xuₜ₊₁, xuₜ, Δt)
        xₜ₊₁ = xuₜ₊₁[1:end-1]
        xₜ = xuₜ[1:end-1]
        uₜ = xuₜ[end:end]
        return dynamics(system, system_integrator, xₜ₊₁, xₜ, uₜ, Δt)
    end

    @views function f(z)
        zs = [z[slice(t, vardim + 1; stretch=-1)] for t in 1:T]
        Δts = [z[index(t, vardim + 1)] for t in 1:T-1]
        δxs = zeros(typeof(z[1]), system.nstates * (T - 1))
        for t = 1:T-1
            δxₜ₊₁ = fₜ(zs[t+1], zs[t], Δts[t])
            δxs[slice(t, system.nstates)] = δxₜ₊₁
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
                    Is .+ (t-1) * system.nstates,
                    Js .+ (t-1) * (vardim + 1)
                )
            ) for t = 1:T-1
        ]...
    )

    ∇f̂ₜ_expr = Symbolics.build_function(∇f̂ₜ_symb, zΔtz)
    ∇f̂ₜ = eval(∇f̂ₜ_expr[1])

    function ∇f(Z)
        jac = spzeros(system.nstates*(T - 1), (vardim + 1)*T)
        for t = 1:T-1
            zₜΔtzₜ₊₁_inds = slice(t, vardim+1; stretch=vardim)
            zₜΔtzₜ₊₁ = Z[zₜΔtzₜ₊₁_inds]
            ∇fₜ = ∇f̂ₜ(zₜΔtzₜ₊₁)
            jac[slice(t, system.nstates), zₜΔtzₜ₊₁_inds] = ∇fₜ
        end
        return jac
    end

    # TODO: redo hessian
    if eval_hessian
        Symbolics.@variables μ[1:system.nstates * (T - 1)]
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
