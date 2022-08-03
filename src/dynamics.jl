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
    xₜ₊₁::AbstractVector,
    xₜ::AbstractVector,
    uₜ::AbstractVector,
    Δt::Real
)

    augsₜ = xₜ[(sys.n_wfn_states + 1):end]
    augsₜ₊₁ = xₜ₊₁[(sys.n_wfn_states + 1):end]

    controlsₜ = [augsₜ[(sys.ncontrols + 1):end]; uₜ]

    #controlsₜ = [augsₜ[1:end]; uₜ]

    δaugs = augsₜ₊₁ - (augsₜ + controlsₜ * Δt)

    δψ̃s = zeros(typeof(xₜ[1]), sys.n_wfn_states)

    #aₜ = augsₜ[slice(2, sys.ncontrols)]
    aₜ = augsₜ[slice(1, sys.ncontrols)]

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
    μ∇²F::Union{Function, Nothing}
    μ∇²F_structure::Union{Vector{Tuple{Int, Int}}, Nothing}
end

function SystemDynamics(
    sys::AbstractQubitSystem,
    integrator::Symbol,
    T::Int,
    Δt::Float64,
    eval_hessian::Bool
)
    sys_integrator = eval(integrator)(sys)

    @views function f(zₜ::AbstractVector, zₜ₊₁::AbstractVector)
        xₜ₊₁ = zₜ₊₁[1:sys.nstates]
        xₜ = zₜ[1:sys.nstates]
        uₜ = zₜ[
            sys.n_wfn_states .+
            slice(sys.augdim + 1, sys.ncontrols)
        ]
        return dynamics(sys, sys_integrator, xₜ₊₁, xₜ, uₜ, Δt)
    end

    @views function F(Z::AbstractVector{F}) where F
        δX = zeros(F, sys.nstates * (T - 1))
        for t = 1:T-1
            zₜ = Z[slice(t, sys.vardim)]
            zₜ₊₁ = Z[slice(t + 1, sys.vardim)]
            δX[slice(t, sys.nstates)] = f(zₜ, zₜ₊₁)
        end
        return δX
    end



    # TODO: figure out a better way to handle sparsity of block matrices

    #
    # ∇ₜf structure list
    #

    ∇ₜf_structure = []


    # ∂ψ̃ⁱₜPⁱ blocks

    for i = 1:sys.nqstates
        for j = 1:sys.isodim     # jth column of ∂ψ̃ⁱₜPⁱ
            for k = 1:sys.isodim # kth row
                kj = (
                    index(i, k, sys.isodim),
                    index(i, j, sys.isodim)
                )
                push!(∇ₜf_structure, kj)
            end
        end
    end


    # ∂aʲₜPⁱ blocks

    for i = 1:sys.nqstates
        for j = 1:sys.ncontrols  # jth column of ∂aʲₜPⁱ
            for k = 1:sys.isodim # kth row of ψ̃ⁱᵏₜ
                kj = (
                    index(i, k, sys.isodim),
                    sys.n_wfn_states + j #+sys.ncontrols
                )
                push!(∇ₜf_structure, kj)
            end
        end
    end


    # -I blocks on main diagonal

    for k = 1:sys.n_aug_states
        kk = (k, k) .+ sys.n_wfn_states
        push!(∇ₜf_structure, kk)
    end


    # -Δt⋅I blocks on shifted diagonal

    for k = 1:sys.n_aug_states
        kk_shifted = (k, k + sys.ncontrols) .+ sys.n_wfn_states
        push!(∇ₜf_structure, kk_shifted)
    end



    #
    # Jacobian of f w.r.t. zₜ (see eq. 7)
    #

    ∇P = Jacobian(sys_integrator)

    @views function ∇ₜf(zₜ::AbstractVector, zₜ₊₁::AbstractVector)

        ∇s = []

        #aₜ = zₜ[sys.n_wfn_states .+ slice(2, sys.ncontrols)]
        aₜ = zₜ[sys.n_wfn_states .+ slice(1, sys.ncontrols)]

        ∂ψ̃ⁱₜPⁱ = ∇P(aₜ, Δt, false)

        for i = 1:sys.nqstates
            append!(∇s, ∂ψ̃ⁱₜPⁱ)
        end

        for i = 1:sys.nqstates

            ψ̃ⁱ_slice = slice(i, sys.isodim)

            for j = 1:sys.ncontrols

                ∂aʲₜPⁱ = ∇P(zₜ₊₁[ψ̃ⁱ_slice], zₜ[ψ̃ⁱ_slice], aₜ, Δt, j)

                append!(∇s, ∂aʲₜPⁱ)
            end
        end

        for _ = 1:sys.n_aug_states
            append!(∇s, -1.0)
        end

        for _ = 1:sys.n_aug_states
            append!(∇s, -Δt)
        end

        return ∇s
    end



    #
    # ∇ₜ₊₁f structure list
    #

    ∇ₜ₊₁f_structure = []


    # ∂ψ̃ⁱₜPⁱ blocks

    for i = 1:sys.nqstates
        for j = 1:sys.isodim     # jth column: ∂ψ̃ⁱʲₜPⁱ
            for k = 1:sys.isodim # kth row: ∂ψ̃ⁱʲₜPⁱᵏ
                kj = (
                    index(i, k, sys.isodim),
                    index(i, j, sys.isodim)
                )
                push!(∇ₜ₊₁f_structure, kj)
            end
        end
    end


    # I for controls on main diagonal

    for k = 1:sys.n_aug_states
        kk = sys.n_wfn_states .+ (k, k)
        push!(∇ₜ₊₁f_structure, kk)
    end



    #
    # Jacobian of f w.r.t. zₜ₊₁ (see eq. 8)
    #

    @views function ∇ₜ₊₁f(zₜ::AbstractVector)

        ∇s = []

        #aₜ = zₜ[sys.n_wfn_states .+ slice(2, sys.ncontrols)]

        aₜ = zₜ[sys.n_wfn_states .+ slice(1, sys.ncontrols)]

        ∂ψ̃ⁱₜ₊₁Pⁱ = ∇P(aₜ, Δt, true)

        for i = 1:sys.nqstates
            append!(∇s, ∂ψ̃ⁱₜ₊₁Pⁱ)
        end

        for _ = 1:sys.n_aug_states
            append!(∇s, 1.0)
        end

        return ∇s
    end


    #
    # full system Jacobian
    #


    # ∇F structure list

    ∇F_structure = []

    for t = 1:T-1

        for (k, j) in ∇ₜf_structure
            kₜ = k + index(t, 0, sys.nstates)
            jₜ = j + index(t, 0, sys.vardim)
            push!(∇F_structure, (kₜ, jₜ))
        end

        for (k, j) in ∇ₜ₊₁f_structure
            kₜ = k + index(t, 0, sys.nstates)
            jₜ = j + sys.vardim + index(t, 0, sys.vardim)
            push!(∇F_structure, (kₜ, jₜ))
        end
    end


    # full Jacobian

    @views function ∇F(Z::AbstractVector)
        ∇s = []
        for t = 1:T-1
            zₜ = Z[slice(t, sys.vardim)]
            zₜ₊₁ = Z[slice(t + 1, sys.vardim)]
            append!(∇s, ∇ₜf(zₜ, zₜ₊₁))
            append!(∇s, ∇ₜ₊₁f(zₜ))
        end
        return ∇s
    end


    #
    # dynamics Hessian
    #

    μ∇²F = nothing
    μ∇²F_structure = nothing

    if eval_hessian

        if sys_integrator isa SecondOrderPade

            μ∇²F_structure = []

            # ∂ψ̃ⁱₜ₍₊₁₎∂aʲₜPⁱ blocks

            for t = 1:T-1
                for i = 1:sys.nqstates

                    # ∂ψ̃ⁱₜ∂aʲₜPⁱ block:

                    for j = 1:sys.ncontrols
                        for k = 1:sys.isodim
                            kⁱjₜ = (
                                # kⁱth row
                                index(t, 0, sys.vardim) +
                                index(i, k, sys.isodim),

                                # jth column
                                index(t, 0, sys.vardim) +
                                sys.n_wfn_states +
                                # FIXME: assumes ∫a exists here
                                #index(2, j, sys.ncontrols)
                                index(1, j, sys.ncontrols)
                            )
                            push!(μ∇²F_structure, kⁱjₜ)
                        end
                    end


                    # ∂aᵏₜ∂ψ̃ⁱₜ₊₁Pⁱ block

                    for k = 1:sys.ncontrols
                        for j = 1:sys.isodim
                            jⁱkₜ₊₁ = (
                                # kth row
                                index(t, 0, sys.vardim) +
                                sys.n_wfn_states +
                                # FIXME: assumes ∫a exists here
                                #index(2, k, sys.ncontrols),
                                index(1, k, sys.ncontrols),

                                # jⁱth column
                                index(t + 1, 0, sys.vardim) +
                                index(i, j, sys.isodim)
                            )
                            push!(μ∇²F_structure, jⁱkₜ₊₁)
                        end
                    end
                end
            end

            H = SecondOrderPadeHessian(sys)

            μ∇²F = @views (
                Z::AbstractVector, # not used here, but keep!
                μ::AbstractVector
            ) -> begin

                Hⁱᵏʲs = []

                for t = 1:T-1
                    μₜ = μ[slice(t, sys.nstates)]
                    for i = 1:sys.nqstates

                        # ∂ψ̃ⁱₜ∂aʲₜPⁱ block
                        for j = 1:sys.ncontrols
                            append!(
                                Hⁱᵏʲs,
                                H(μₜ, Δt, i, j, false)
                            )
                        end

                        # ∂ψ̃ⁱₜ₊₁∂aʲₜPⁱ block
                        for k = 1:sys.ncontrols
                            append!(
                                Hⁱᵏʲs,
                                H(μₜ, Δt, i, k, true)
                            )
                        end
                    end
                end
                return Hⁱᵏʲs
            end

        elseif sys_integrator isa FourthOrderPade

            μ∇²F_structure = []

            for t = 1:T-1

                # ∂aᵏₜ∂aʲₜPⁱ block

                for j = 1:sys.ncontrols
                    for k = 1:j
                        kjₜ = (
                            index(t, 0, sys.vardim) +
                            sys.n_wfn_states +
                            # FIXME: assumes ∫a exists here
                            #index(2, k, sys.ncontrols),
                            index(1, k, sys.ncontrols),

                            index(t, 0, sys.vardim) +
                            sys.n_wfn_states +
                            # FIXME: assumes ∫a exists here
                            # index(2, j, sys.ncontrols),
                            index(1, j, sys.ncontrols)
                        )
                        push!(μ∇²F_structure, kjₜ)
                    end
                end


                # ∂ψ̃ⁱₜ₍₊₁₎∂aʲₜPⁱ blocks

                for i = 1:sys.nqstates

                    # ∂ψ̃ⁱₜ∂aʲₜPⁱ block:

                    for j = 1:sys.ncontrols
                        for k = 1:sys.isodim
                            kⁱjₜ = (
                                # kⁱth row
                                index(t, 0, sys.vardim) +
                                index(i, k, sys.isodim),

                                # jth column
                                index(t, 0, sys.vardim) +
                                sys.n_wfn_states +
                                # FIXME: assumes ∫a exists here
                                #index(2, j, sys.ncontrols)
                                index(1, j, sys.ncontrols)
                            )
                            push!(μ∇²F_structure, kⁱjₜ)
                        end
                    end


                    # ∂aᵏₜ∂ψ̃ⁱₜ₊₁Pⁱ block

                    for k = 1:sys.ncontrols
                        for j = 1:sys.isodim
                            jⁱkₜ₊₁ = (
                                # kth row
                                index(t, 0, sys.vardim) +
                                sys.n_wfn_states +
                                # FIXME: assumes ∫a exists here
                                #index(2, k, sys.ncontrols),
                                index(1, k, sys.ncontrols),

                                # jⁱth column
                                index(t + 1, 0, sys.vardim) +
                                index(i, j, sys.isodim)
                            )
                            push!(μ∇²F_structure, jⁱkₜ₊₁)
                        end
                    end
                end
            end

            H = FourthOrderPadeHessian(sys)

            μ∇²F = @views (
                Z::AbstractVector,
                μ::AbstractVector
            ) -> begin

                Hs = []

                for t = 1:T-1

                    ψ̃ₜ₊₁ = Z[
                        slice(
                            t + 1,
                            sys.n_wfn_states,
                            sys.vardim
                        )
                    ]

                    ψ̃ₜ = Z[
                        slice(
                            t,
                            sys.n_wfn_states,
                            sys.vardim
                        )
                    ]

                    # dim(μ) = sys.nstates * (T - 1)
                    μₜ = μ[slice(t, sys.nstates)]

                    # ∂aᵏₜ∂aʲₜPⁱ block

                    for j = 1:sys.ncontrols
                        for k = 1:j
                            Hᵏʲ = H(ψ̃ₜ₊₁, ψ̃ₜ, μₜ, Δt, k, j)
                            append!(Hs, Hᵏʲ)
                        end
                    end

                    aₜ = Z[
                        index(
                            t,
                            sys.n_wfn_states,
                            sys.vardim
                        ) .+

                        # FIXME: assumes ∫a exists here
                        # slice(
                        #     2,
                        #     sys.ncontrols
                        # )
                        slice(1, sys.ncontrols)
                    ]

                    # ∂ψ̃ⁱₜ₍₊₁₎∂aʲₜPⁱ blocks

                    for i = 1:sys.nqstates

                        # ∂ψ̃ⁱₜ∂aʲₜPⁱ block
                        for j = 1:sys.ncontrols
                            append!(
                                Hs,
                                H(aₜ, μₜ, Δt, i, j, false)
                            )
                        end

                        # ∂ψ̃ⁱₜ₊₁∂aʲₜPⁱ block
                        for k = 1:sys.ncontrols
                            append!(
                                Hs,
                                H(aₜ, μₜ, Δt, i, k, true)
                            )
                        end
                    end
                end
                return Hs
            end
        end
    end

    return SystemDynamics(
        F,
        ∇F,
        ∇F_structure,
        μ∇²F,
        μ∇²F_structure
    )
end


#
#
# min time sys dynamics (dynamical Δts)
#
#

# TODO: reimplement

function MinTimeSystemDynamics(
    sys::AbstractQubitSystem,
    integrator::Symbol,
    T::Int,
    eval_hessian::Bool
)

    sys_integrator = eval(integrator)(sys)

    @views function f(
        zₜ::AbstractVector,
        zₜ₊₁::AbstractVector,
        Δtₜ::Real
    )
        xₜ₊₁ = zₜ₊₁[1:sys.nstates]
        xₜ = zₜ[1:sys.nstates]
        uₜ = zₜ[
            sys.n_wfn_states .+
            slice(sys.augdim + 1, sys.ncontrols)
        ]
        return dynamics(sys, sys_integrator, xₜ₊₁, xₜ, uₜ, Δtₜ)
    end

    @views function F(Z::AbstractVector{F}) where F
        δX = zeros(F, sys.nstates * (T - 1))
        for t = 1:T-1
            zₜ = Z[slice(t, sys.vardim)]
            zₜ₊₁ = Z[slice(t + 1, sys.vardim)]
            Δtₜ = Z[end - (T - 1) + t]
            δX[slice(t, sys.nstates)] = f(zₜ, zₜ₊₁, Δtₜ)
        end
        return δX
    end

    #
    # ∇ₜf structure list
    #

    ∇ₜf_structure = []


    # ∂ψ̃ⁱₜPⁱ blocks

    for i = 1:sys.nqstates
        for j = 1:sys.isodim     # jth column of ∂ψ̃ⁱₜPⁱ
            for k = 1:sys.isodim # kth row
                kj = (
                    index(i, k, sys.isodim),
                    index(i, j, sys.isodim)
                )
                push!(∇ₜf_structure, kj)
            end
        end
    end


    # ∂aʲₜPⁱ blocks

    for i = 1:sys.nqstates
        for j = 1:sys.ncontrols  # jth column of ∂aʲₜPⁱ
            for k = 1:sys.isodim # kth row of ψ̃ⁱᵏₜ
                kj = (
                    index(i, k, sys.isodim),
                    sys.n_wfn_states + sys.ncontrols + j
                )
                push!(∇ₜf_structure, kj)
            end
        end
    end


    # -I blocks on main diagonal

    for k = 1:sys.n_aug_states
        kk = (k, k) .+ sys.n_wfn_states
        push!(∇ₜf_structure, kk)
    end


    # -Δt⋅I blocks on shifted diagonal

    for k = 1:sys.n_aug_states
        kk_shifted = (k, k + sys.ncontrols) .+ sys.n_wfn_states
        push!(∇ₜf_structure, kk_shifted)
    end



    #
    # Jacobian of f w.r.t. zₜ (see eq. 7)
    #

    ∇P = Jacobian(sys_integrator)

    @views function ∇ₜf(
        zₜ::AbstractVector,
        zₜ₊₁::AbstractVector,
        Δtₜ::Real
    )

        ∇s = []

        aₜ = zₜ[sys.n_wfn_states .+ slice(2, sys.ncontrols)]

        ∂ψ̃ⁱₜPⁱ = ∇P(aₜ, Δtₜ, false)

        for i = 1:sys.nqstates
            append!(∇s, ∂ψ̃ⁱₜPⁱ)
        end

        for i = 1:sys.nqstates

            ψ̃ⁱ_slice = slice(i, sys.isodim)

            for j = 1:sys.ncontrols

                ∂aʲₜPⁱ = ∇P(
                    zₜ₊₁[ψ̃ⁱ_slice],
                    zₜ[ψ̃ⁱ_slice],
                    aₜ,
                    Δtₜ,
                    j
                )

                append!(∇s, ∂aʲₜPⁱ)
            end
        end

        for _ = 1:sys.n_aug_states
            append!(∇s, -1.0)
        end

        for _ = 1:sys.n_aug_states
            append!(∇s, -Δtₜ)
        end

        return ∇s
    end



    #
    # ∇ₜ₊₁f structure list
    #

    ∇ₜ₊₁f_structure = []


    # ∂ψ̃ⁱₜPⁱ blocks

    for i = 1:sys.nqstates
        for j = 1:sys.isodim     # jth column: ∂ψ̃ⁱʲₜPⁱ
            for k = 1:sys.isodim # kth row: ∂ψ̃ⁱʲₜPⁱᵏ
                kj = (
                    index(i, k, sys.isodim),
                    index(i, j, sys.isodim)
                )
                push!(∇ₜ₊₁f_structure, kj)
            end
        end
    end


    # I for controls on main diagonal

    for k = 1:sys.n_aug_states
        kk = sys.n_wfn_states .+ (k, k)
        push!(∇ₜ₊₁f_structure, kk)
    end



    #
    # Jacobian of f w.r.t. zₜ₊₁ (see eq. 8)
    #

    @views function ∇ₜ₊₁f(zₜ::AbstractVector, Δtₜ::Real)

        ∇s = []

        aₜ = zₜ[sys.n_wfn_states .+ slice(2, sys.ncontrols)]

        ∂ψ̃ⁱₜ₊₁Pⁱ = ∇P(aₜ, Δtₜ, true)

        for i = 1:sys.nqstates
            append!(∇s, ∂ψ̃ⁱₜ₊₁Pⁱ)
        end

        for _ = 1:sys.n_aug_states
            append!(∇s, 1.0)
        end

        return ∇s
    end


    #
    # full system Jacobian
    #


    # ∇F structure list

    ∇F_structure = []

    for t = 1:T-1

        for (k, j) in ∇ₜf_structure
            kₜ = k + index(t, 0, sys.nstates)
            jₜ = j + index(t, 0, sys.vardim)
            push!(∇F_structure, (kₜ, jₜ))
        end

        for (k, j) in ∇ₜ₊₁f_structure
            kₜ = k + index(t, 0, sys.nstates)
            jₜ = j + sys.vardim + index(t, 0, sys.vardim)
            push!(∇F_structure, (kₜ, jₜ))
        end
    end


    # full Jacobian

    @views function ∇F(Z::AbstractVector)
        ∇s = []
        for t = 1:T-1
            zₜ = Z[slice(t, sys.vardim)]
            zₜ₊₁ = Z[slice(t + 1, sys.vardim)]
            Δtₜ = Z[end - (T - 1) + t]
            append!(∇s, ∇ₜf(zₜ, zₜ₊₁, Δtₜ))
            append!(∇s, ∇ₜ₊₁f(zₜ, Δtₜ))
        end
        return ∇s
    end


    #
    # dynamics Hessian
    #

    μ∇²F = nothing
    μ∇²F_structure = nothing

    if eval_hessian

        if sys_integrator isa SecondOrderPade

            μ∇²F_structure = []


            for t = 1:T-1
                for i = 1:sys.nqstates

                    # ∂ψ̃ⁱₜ∂aʲₜPⁱ block:

                    for j = 1:sys.ncontrols
                        for k = 1:sys.isodim
                            kⁱjₜ = (
                                # kⁱth row
                                index(t, 0, sys.vardim) +
                                index(i, k, sys.isodim),

                                # jth column
                                index(t, 0, sys.vardim) +
                                sys.n_wfn_states +
                                # FIXME: assumes ∫a exists here
                                #index(2, j, sys.ncontrols)
                                index(1, j, sys.ncontrols)
                            )
                            push!(μ∇²F_structure, kⁱjₜ)
                        end
                    end


                    # ∂aᵏₜ∂ψ̃ⁱₜ₊₁Pⁱ block

                    for k = 1:sys.ncontrols
                        for j = 1:sys.isodim
                            jⁱkₜ₊₁ = (
                                # kth row
                                index(t, 0, sys.vardim) +
                                sys.n_wfn_states +
                                # FIXME: assumes ∫a exists here
                                # index(2, k, sys.ncontrols),
                                index(1, k, sys.ncontrols),

                                # jⁱth column
                                index(t + 1, 0, sys.vardim) +
                                index(i, j, sys.isodim)
                            )
                            push!(μ∇²F_structure, jⁱkₜ₊₁)
                        end
                    end
                end
            end


            H = SecondOrderPadeHessian(sys)

            μ∇²F = @views (
                Z::AbstractVector,
                μ::AbstractVector
            ) -> begin
                Hⁱᵏʲs = []

                for t = 1:T-1
                    μₜ = μ[slice(t, sys.nstates)]
                    Δtₜ = Z[end - (T - 1) + t]
                    for i = 1:sys.nqstates

                        # ∂ψ̃ⁱₜ∂aʲₜPⁱ block
                        for j = 1:sys.ncontrols
                            append!(
                                Hⁱᵏʲs,
                                H(μₜ, Δtₜ, i, j, false)
                            )
                        end

                        # ∂ψ̃ⁱₜ₊₁∂aʲₜPⁱ block
                        for k = 1:sys.ncontrols
                            append!(
                                Hⁱᵏʲs,
                                H(μₜ, Δtₜ, i, k, true)
                            )
                        end
                    end
                end
                return Hⁱᵏʲs
            end

        elseif sys_integrator isa FourthOrderPade

            μ∇²F_structure = []

            for t = 1:T-1

                # ∂aᵏₜ∂aʲₜPⁱ block

                for j = 1:sys.ncontrols
                    for k = 1:j
                        kjₜ = (
                            index(t, 0, sys.vardim) +
                            sys.n_wfn_states +
                            # FIXME: assumes ∫a exists here
                            #index(2, k, sys.ncontrols),
                            index(1, k, sys.ncontrols),

                            index(t, 0, sys.vardim) +
                            sys.n_wfn_states +
                            # FIXME: assumes ∫a exists here
                            #index(2, j, sys.ncontrols),
                            index(1, k, sys.controls)
                        )
                        push!(μ∇²F_structure, kjₜ)
                    end
                end


                # ∂ψ̃ⁱₜ₍₊₁₎∂aʲₜPⁱ blocks

                for i = 1:sys.nqstates

                    # ∂ψ̃ⁱₜ∂aʲₜPⁱ block:

                    for j = 1:sys.ncontrols
                        for k = 1:sys.isodim
                            kⁱjₜ = (
                                # kⁱth row
                                index(t, 0, sys.vardim) +
                                index(i, k, sys.isodim),

                                # jth column
                                index(t, 0, sys.vardim) +
                                sys.n_wfn_states +
                                # FIXME: assumes ∫a exists here
                                #index(2, j, sys.ncontrols)
                                index(1, j, sys.ncontrols)
                            )
                            push!(μ∇²F_structure, kⁱjₜ)
                        end
                    end


                    # ∂aᵏₜ∂ψ̃ⁱₜ₊₁Pⁱ block

                    for k = 1:sys.ncontrols
                        for j = 1:sys.isodim
                            jⁱkₜ₊₁ = (
                                # kth row
                                index(t, 0, sys.vardim) +
                                sys.n_wfn_states +
                                # FIXME: assumes ∫a exists here
                                #index(2, k, sys.ncontrols),
                                index(1, k, sys.ncontrols),

                                # jⁱth column
                                index(t + 1, 0, sys.vardim) +
                                index(i, j, sys.isodim)
                            )
                            push!(μ∇²F_structure, jⁱkₜ₊₁)
                        end
                    end
                end
            end

            H = FourthOrderPadeHessian(sys)

            μ∇²F = @views (
                Z::AbstractVector,
                μ::AbstractVector
            ) -> begin
                Hs = []
                for t = 1:T-1

                    ψ̃ₜ₊₁ = Z[
                        slice(
                            t + 1,
                            sys.n_wfn_states,
                            sys.vardim
                        )
                    ]

                    ψ̃ₜ = Z[
                        slice(
                            t,
                            sys.n_wfn_states,
                            sys.vardim
                        )
                    ]

                    # dim(μ) = sys.nstates * (T - 1)
                    μₜ = μ[slice(t, sys.nstates)]

                    Δtₜ = Z[end - (T - 1) + t]


                    # ∂aᵏₜ∂aʲₜPⁱ block

                    for j = 1:sys.ncontrols
                        for k = 1:j
                            Hᵏʲ = H(ψ̃ₜ₊₁, ψ̃ₜ, μₜ, Δtₜ, k, j)
                            append!(Hs, Hᵏʲ)
                        end
                    end

                    aₜ = Z[
                        index(
                            t,
                            sys.n_wfn_states,
                            sys.vardim
                        ) .+

                        # FIXME: assumes ∫a exists here
                        # slice(
                        #     2,
                        #     sys.ncontrols
                        # )
                        slice(1, sys.ncontrols)
                    ]

                    # ∂ψ̃ⁱₜ₍₊₁₎∂aʲₜPⁱ blocks

                    for i = 1:sys.nqstates

                        # ∂ψ̃ⁱₜ∂aʲₜPⁱ block
                        for j = 1:sys.ncontrols
                            append!(
                                Hs,
                                H(aₜ, μₜ, Δtₜ, i, j, false)
                            )
                        end

                        # ∂ψ̃ⁱₜ₊₁∂aʲₜPⁱ block
                        for k = 1:sys.ncontrols
                            append!(
                                Hs,
                                H(aₜ, μₜ, Δtₜ, i, k, true)
                            )
                        end
                    end
                end
                return Hs
            end
        end
    end

    return SystemDynamics(
        F,
        ∇F,
        ∇F_structure,
        μ∇²F,
        μ∇²F_structure
    )
end


end
