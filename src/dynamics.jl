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
    aₜ = xₜ[end-system.control_order+1]
    aₜs = xₜ[(end-system.control_order):end]
    aₜ₊₁s = xₜ₊₁[(end-system.control_order):end]
    âₜ₊₁s = zeros(typeof(xₜ[1]), system.control_order + 1)
    for i = 1:system.control_order
        âₜ₊₁s[i] = aₜs[i] + Δt * aₜs[i+1]
    end
    âₜ₊₁s[end] = aₜs[end] + Δt * uₜ[1]
    δaₜ₊₁s = aₜ₊₁s - âₜ₊₁s
    ψ̃ₜs = [xₜ[slice(i, system.isodim)] for i = 1:system.nqstates]
    ψ̃ₜ₊₁s = [xₜ₊₁[slice(i, system.isodim)] for i = 1:system.nqstates]
    δψ̃ₜ₊₁s = [integrator(ψ̃ₜ₊₁s[i], ψ̃ₜs[i], aₜ) for i = 1:system.nqstates]
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
    eval_hessian::Bool,
    T::Int,
    Δt::Float64
) where N

    vardim = system.nstates + 1

    function system_integrator(ψ̃ₜ₊₁, ψ̃ₜ, aₜ)
        return integrator(ψ̃ₜ₊₁, ψ̃ₜ, aₜ, Δt, system.G_drift, system.G_drive)
    end

    @views function fₜ(xuₜ₊₁, xuₜ)
        xₜ₊₁ = xuₜ₊₁[1:end-1]
        xₜ = xuₜ[1:end-1]
        uₜ = xuₜ[end:end]
        return dynamics(system, system_integrator, xₜ₊₁, xₜ, uₜ, Δt)
    end

    @views function f(y)
        xus = [y[slice(t, vardim)] for t in 1:T]
        δxs = zeros(typeof(y[1]), system.nstates * (T - 1))
        for t = 1:T-1
            δxₜ₊₁ = fₜ(xus[t+1], xus[t])
            δxs[slice(t, system.nstates)] = δxₜ₊₁
        end
        return δxs
    end

    Symbolics.@variables z[1:2*vardim]

    z = collect(z)

    f̂ₜ(xuₜxuₜ₊₁) = fₜ(xuₜxuₜ₊₁[vardim+1:end], xuₜxuₜ₊₁[1:vardim])

    ∇f̂ₜ_symb = Symbolics.sparsejacobian(f̂ₜ(z), z)

    Is = findnz(∇f̂ₜ_symb)[1]
    Js = findnz(∇f̂ₜ_symb)[2]

    ∇f_structure = vcat(
        [
            collect(
                zip(
                    Is .+ (t-1) * system.nstates,
                    Js .+ (t-1) * vardim
                )
            ) for t = 1:T-1
        ]...
    )

    ∇f̂ₜ_expr = Symbolics.build_function(∇f̂ₜ_symb, z)
    ∇f̂ₜ = eval(∇f̂ₜ_expr[1])

    function ∇f(y)
        xus = [y[slice(t, vardim)] for t in 1:T]
        jac = spzeros(system.nstates*(T-1), vardim*T)
        for t = 1:T-1
            ∇fₜ = ∇f̂ₜ([xus[t]; xus[t+1]])
            jac[slice(t, system.nstates), slice(t, vardim; stretch=vardim)] = ∇fₜ
        end
        return jac
    end

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
