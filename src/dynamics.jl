module Dynamics

export SystemDynamics
export MinTimeSystemDynamics

using ..Utils
using ..QuantumLogic
using ..QubitSystems
using ..Integrators

using LinearAlgebra
using SparseArrays

#
# dynamics functions
#

@views function dynamics(
    sys::AbstractQubitSystem,
    integrator::AbstractQuantumIntegrator,
    xₜ₊₁::Vector,
    xₜ::Vector,
    uₜ::Vector,
    Δt::Real
)

    augsₜ = xₜ[(sys.n_wfn_states + 1):end]
    augsₜ₊₁ = xₜ₊₁[(sys.n_wfn_states + 1):end]

    controlsₜ = [augsₜ[(sys.ncontrols + 1):end]; uₜ]

    δaugs = augsₜ₊₁ - (augsₜ + controlsₜ * Δt)

    δψ̃s = zeros(typeof(xₜ[1]), sys.n_wfn_states)

    aₜ = augsₜ[slice(2, sys.ncontrols)]

    for i = 1:sys.nqstates
        ψ̃ⁱslice = slice(i, sys.isodim)
        δψ̃s[ψ̃ⁱslice] = integrator(xₜ₊₁[ψ̃ⁱslice], xₜ[ψ̃ⁱslice], aₜ, Δt)
    end

    return [δψ̃s; δaugs]
end

struct SystemDynamics
    F::Function
    ∇F::Function
    ∇F_structure::Vector{Tuple{Int, Int}}
    ∇²F::Union{Function, Nothin}
    ∇²F_structrure::Union{Vector{Tuple{Int, Int}}, Nothing}
end

function SystemDynamics(
    sys::AbstractQubitSystem,
    integrator::AbstractQuantumIntegrator,
    T::Int,
    Δt::Float64,
    eval_hessian::Bool
)
    sys_integrator = integrator(sys)

    @views function f(zₜ, zₜ₊₁)
        xₜ₊₁ = zₜ₊₁[1:sys.nstates]
        xₜ = zₜ[1:sys.nstates]
        uₜ = zₜ[(end - sys.ncontrols + 1):end]
        return dynamics(sys, sys_integrator, xₜ₊₁, xₜ, uₜ, Δt)
    end

    @views function F(Z)
        δxs = zeros(typeof(Z[1]), sys.nstates * (T - 1))
        for t = 1:T-1
            δxₜ = f(Z[slice(t, sys.vardim)], Z[slice(t + 1, sys.vardim)])
            δxs[slice(t - 1, sys.nstates)] = δxₜ
        end
        return δxs
    end


    # TODO: figure out a better way to handle sparsity of block matrices

    #
    # ∇ₜf structure list
    #

    ∇ₜf_structure = []


    # ∂Pⁱ_∂ψ̃ⁱₜ blocks

    for i = 1:sys.nqstates
        for j = 1:sys.isodim     # jth column of ∂Pⁱ_∂ψ̃ⁱₜ
            for k = 1:sys.isodim # kth row
                idx = (index(i, k, sys.isodim), index(i, j, sys.isodim))
                append!(∇ₜf_structure, idx)
            end
        end
    end


    # ∂Pⁱ_∂aʲₜ blocks

    for i = 1:sys.nqstates
        for j = 1:sys.ncontrols  # jth column, corresponding to ∂Pⁱ_∂aʲₜ
            for k = 1:sys.isodim # kth row, corresponding to ψ̃ⁱₜ
                idx = (index(i, k, sys.isodim), sys.n_wfn_states + sys.ncontrols + j)
                append!(∇ₜf_structure, idx)
            end
        end
    end


    # -I blocks on main diagonal

    for k = 1:sys.n_aug_states
        idx = (k, k) .+ sys.n_wfn_states
        append!(∇ₜf_structure, idx)
    end


    # -Δt⋅I blocks on shifted diagonal

    for k = 1:sys.n_aug_states
        idx = (k, k + sys.ncontrols) .+ sys.n_wfn_states
        append!(∇ₜf_structure, idx)
    end


    #
    # Jacobian of f w.r.t. zₜ (see eq. 7)
    #

    ∇P = Jacobian(integrator)

    @views function ∇ₜf(zₜ::Vector, zₜ₊₁::Vector)
        Js = []

        a_slice = sys.n_wfn_states .+ slice(2, sys.ncontrols)
        aₜ = zₜ[a_slice]

        for i = 1:sys.nqstates

            ∂ψ̃ⁱₜPⁱ = ∇P(aₜ, Δt, false)

            for ∂ψ̃ⁱʲₜPⁱᵏ in ∂ψ̃ⁱₜPⁱ
                append!(Js, ∂ψ̃ⁱʲₜPⁱᵏ)
            end

            ∂aₜPⁱ = zeros(sys.n_wfn_states, sys.ncontrols)

            ψ̃ⁱ_slice = slice(i, sys.isodim)

            for j = 1:sys.ncontrols

                ∂aʲₜPⁱ = ∇P(zₜ₊₁[ψ̃ⁱ_slice], zₜ[ψ̃ⁱ_slice], aₜ, Δt, j)

                ∂aₜPⁱ[:, j] = ∂aʲₜPⁱ
            end

            for ∂aʲₜPⁱᵏ in ∂aₜPⁱ
                append!(Js, ∂aʲₜPⁱᵏ)
            end
        end

        for _ = 1:sys.n_aug_states
            append!(Js, -1.0)
        end

        for _ = 1:sys.n_aug_states
            append!(Js, -Δt)
        end

        return Js
    end


    #
    # ∇ₜ₊₁f structure list
    #

    ∇ₜ₊₁f_structure = []


    # ∂Pⁱ_∂ψ̃ⁱₜ blocks

    for i = 1:sys.nqstates
        for j = 1:sys.isodim     # jth column of ∂Pⁱ_∂ψ̃ⁱₜ
            for k = 1:sys.isodim # kth row
                idx = (index(i, k, sys.isodim), index(i, j, sys.isodim))
                append!(∇ₜ₊₁f_structure, idx)
            end
        end
    end


    # I for controls on main diagonal

    for k = 1:sys.n_aug_states
        idx = sys.n_wfn_states .+ (k, k)
        append!(∇ₜ₊₁f_structure, idx)
    end


    #
    # Jacobian of f w.r.t. zₜ₊₁ (see eq. 8)
    #

    @views function ∇ₜ₊₁f(zₜ::Vector)
        Js = []

        aₜ = zₜ[sys.n_wfn_states .+ slice(2, sys.ncontrols)]

        for i = 1:sys.nqstates

            ∂ψ̃ⁱₜ₊₁Pⁱ = ∇P(aₜ, Δt, true)

            for ∂ψ̃ⁱʲₜ₊₁Pⁱᵏ in ∂ψ̃ⁱₜ₊₁Pⁱ
                append!(Js, ∂ψ̃ⁱʲₜ₊₁Pⁱᵏ)
            end

            ψ̃ⁱ_slice = slice(i, sys.isodim)

            J[ψ̃ⁱ_slice, ψ̃ⁱ_slice] = ∂Pⁱ_∂ψ̃ⁱₜ₊₁
        end

        for _ = 1:sys.n_aug_states
            append!(Js, 1.0)
        end

        return Js
    end


    #
    # full system Jacobian
    #


    # ∇F structure list

    ∇F_structure = []

    for t = 1:T-1

        for (i, j) in ∇ₜf_structure
            i += index(t, 0, sys.nstates)
            j += index(t, 0, sys.vardim)
            append!(∇F_structure, (i, j))
        end

        for (i, j) in ∇ₜ₊₁f_structure
            i += index(t, 0, sys.nstates)
            j += sys.vardim + index(t, 0, sys.vardim)
            append!(∇F_structure, (i, j))
        end
    end


    # full Jacobian

    @views function ∇F(Z::Vector)
        Js = []
        for t = 1:T-1
            zₜ = Z[slice(t, sys.vardim)]
            zₜ₊₁ = Z[slice(t + 1, sys.vardim)]
            append!(Js, ∇ₜf(zₜ, zₜ₊₁))
            append!(Js, ∇ₜ₊₁f(zₜ))
        end
        return Js
    end


    #
    # dynamics Hessian
    #

    if eval_hessian

        if isa(integrator, FourthOrderPade)

            ∇²F_structure = []

            for t = 1:T-1
                for j = 1:sys.ncontrols
                    for k = 1:j
                        idx = (
                            index(t, sys.n_wfn_states + sys.ncontrols + k),
                            index(t, sys.n_wfn_states + sys.ncontrols + j)
                        )
                        append!(∇²F_structure, idx)
                    end
                end
            end

            H = FourthOrderPadeHessian(sys)

            function ∇²F(Z::Vector, μ::Vector)
                Hᵏʲs = []
                for t = 1:T-1
                    ψ̃ₜ₊₁ = Z[slice(t, 1, sys.n_wfn_states, sys.vardim)]
                    ψ̃ₜ = Z[slice(t, 1, sys.n_wfn_states, sys.vardim)]
                    μₜ = μ[slice(t, sys.nstates)] # pretty sure dim(μ) = nstates * (T - 1)
                    for j = 1:sys.ncontrols
                        for k = 1:j
                            Hᵏʲ = H(ψ̃ₜ₊₁, ψ̃ₜ, μₜ, Δt, k, j)
                            append!(Hᵏʲs, Hᵏʲ)
                        end
                    end
                end
            end
        else

            ∇²F = Z -> []
            ∇²F_structure = []
        end
    else

        ∇²F = nothing
        ∇²F_structure = nothing
    end

    return SystemDynamics(f, ∇F, ∇F_structure, ∇²F, ∇²F_structure)
end



#
#
# min time sys dynamics (dynamical Δts)
#
#

# TODO: reimplement

function MinTimeSystemDynamics(
    sys::AbstractQubitSystem,
    integrator::AbstractQuantumIntegrator,
    T::Int,
    eval_hessian::Bool
)

    sys_integrator = integrator(sys)

    @views function fₜ(xuₜ₊₁, xuₜ, Δt)
        xₜ₊₁ = xuₜ₊₁[1:sys.nstates]
        xₜ = xuₜ[1:sys.nstates]
        uₜ = xuₜ[end - sys.ncontrols + 1:end]
        return dynamics(sys, sys_integrator, xₜ₊₁, xₜ, uₜ, Δt)
    end

    @views function f(z)
        zs = [z[slice(t, 1, sys.vardim, sys.vardim + 1)] for t in 1:T]
        Δts = [z[index(t, sys.vardim + 1)] for t in 1:T-1]
        δxs = zeros(typeof(z[1]), sys.nstates * (T - 1))
        for t = 1:T-1
            δxₜ₊₁ = fₜ(zs[t+1], zs[t], Δts[t])
            δxs[slice(t, sys.nstates)] = δxₜ₊₁
        end
        return δxs
    end

    Symbolics.@variables zΔtz[1:(2*sys.vardim + 1)]

    zΔtz = collect(zΔtz)

    @views f̂ₜ(zₜΔtzₜ₊₁) = fₜ(
        zₜΔtzₜ₊₁[(end - sys.vardim + 1):end], # zₜ₊₁
        zₜΔtzₜ₊₁[1:sys.vardim],               # zₜ
        zₜΔtzₜ₊₁[sys.vardim + 1],             # Δt
    )

    ∇f̂ₜ_symb = Symbolics.sparsejacobian(f̂ₜ(zΔtz), zΔtz)

    Is = findnz(∇f̂ₜ_symb)[1]
    Js = findnz(∇f̂ₜ_symb)[2]

    ∇f_structure = []

    for t = 1:T-1
        idxs = zip(
            Is .+ index(t, 0, sys.nstates),
            Js .+ index(t, 0, sys.vardim + 1)
        )
        append!(∇f_structure, collect(idxs))
    end

    ∇f̂ₜ_expr = Symbolics.build_function(∇f̂ₜ_symb, zΔtz)
    ∇f̂ₜ = eval(∇f̂ₜ_expr[1])

    function ∇f(Z)
        jac = spzeros(sys.nstates * (T - 1), (sys.vardim + 1) * T)
        for t = 1:T-1
            zₜΔtzₜ₊₁_idxs = slice(t, sys.vardim + 1; stretch=sys.vardim)
            zₜΔtzₜ₊₁ = Z[zₜΔtzₜ₊₁_idxs]
            ∇fₜ = ∇f̂ₜ(zₜΔtzₜ₊₁)
            jac[slice(t, sys.nstates), zₜΔtzₜ₊₁_idxs] = ∇fₜ
        end
        return jac
    end

    return SystemDynamics(f, ∇f, ∇f_structure, ∇²f, ∇²f_structure)
end


end
