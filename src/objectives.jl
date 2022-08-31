module Objectives

export QuantumObjective
export MinTimeObjective

using ..Utils
using ..QuantumSystems
using ..Costs

using LinearAlgebra
using SparseArrays
using Symbolics

#
# objective functions
#

abstract type AbstractObjective end

struct QuantumObjective <: AbstractObjective
    L::Function
    ∇L::Function
    ∇²L::Union{Function, Nothing}
    ∇²L_structure::Union{Vector{Tuple{Int,Int}}, Nothing}
end

function QuantumObjective(
    system::AbstractQuantumSystem,
    cost_fn::Symbol,
    T::Int,
    Q::Float64,
    R::Float64,
    eval_hessian::Bool
)
    cost = QuantumCost(system, cost_fn)

    @views function L(Z::AbstractVector{F}) where F
        ψ̃T = Z[slice(T, system.n_wfn_states, system.vardim)]
        obj = 0.0
        for t = 1:T-1
            uₜ = Z[slice(
                t,
                system.nstates + 1,
                system.vardim,
                system.vardim
            )]
            obj += R / 2 * sum(uₜ.^2)
        end
        return Q * cost(ψ̃T) + obj
    end

    ∇c = QuantumCostGradient(cost)

    @views function ∇L(Z::AbstractVector{F}) where F
        ∇ = zeros(F, system.vardim * T)

        for t = 1:T-1
            uₜ_slice = slice(
                t,
                system.nstates + 1,
                system.vardim,
                system.vardim
            )
            uₜ = Z[uₜ_slice]
            ∇[uₜ_slice] = R * uₜ
        end

        ψ̃T_slice = slice(T, system.n_wfn_states, system.vardim)
        ψ̃T = Z[ψ̃T_slice]
        ∇[ψ̃T_slice] = Q * ∇c(ψ̃T)

        return ∇
    end

    ∇²L = nothing
    ∇²L_structure = nothing

    if eval_hessian
        ∇²c = QuantumCostHessian(cost)

        ∇²L_structure = []

        # uₜ Hessian structure (eq. 17)
        for t = 1:T-1
            uₜ_slice = slice(
                t,
                system.nstates + 1,
                system.vardim,
                system.vardim
            )
            append!(
                ∇²L_structure,
                collect(zip(uₜ_slice, uₜ_slice))
            )
        end

        # ℓⁱs Hessian structure (eq. 17)
        append!(
            ∇²L_structure,
            structure(∇²c, T, system.vardim)
        )

        ∇²L = Z::AbstractVector -> begin
            Hs = fill(R, system.ncontrols * (T - 1))
            ψ̃T = view(
                Z,
                slice(T, system.n_wfn_states, system.vardim)
            )
            append!(Hs, Q * ∇²c(ψ̃T))
            return Hs
        end
    end

    return QuantumObjective(L, ∇L, ∇²L, ∇²L_structure)
end

abstract type AbstractRegularizer end

struct MultiModeRegularizer <: AbstractRegularizer
    L::Function
    ∇L::Function
    ∇²L::Union{Function, Nothing}
    ∇²L_structure::Union{Vector{Tuple{Int,Int}}, Nothing}
end

function MultiModeRegularizer(

)
end




#
# min time objective hessian
#

struct MinTimeObjective
    L::Function
    ∇L::Function
    ∇²L::Union{Function, Nothing}
    ∇²L_structure::Union{Vector{Tuple{Int,Int}}, Nothing}
end

function MinTimeObjective(
    sys::AbstractQuantumSystem,
    T::Int,
    Rᵤ::Float64,
    Rₛ::Float64,
    eval_hessian::Bool
)

    total_time(Z::AbstractVector) =
        sum(Z[(end - (T - 1) + 1):end])

    @views function u_smoothness_regulator(Z::AbstractVector)
        ∑Δu² = 0.0

        for t = 1:T-2

            uₜ₊₁ = Z[
                slice(
                    t + 1,
                    sys.nstates + 1,
                    sys.vardim,
                    sys.vardim
                )
            ]

            uₜ = Z[
                slice(
                    t,
                    sys.nstates + 1,
                    sys.vardim,
                    sys.vardim
                )
            ]

            Δu = uₜ₊₁ - uₜ

            ∑Δu² += dot(Δu, Δu)
        end

        return 0.5 * Rₛ * ∑Δu²
    end

    @views function u_amplitude_regulator(Z::AbstractVector)
        ∑u² = 0.0
        for t = 1:T-1
            uₜ = Z[
                slice(
                    t,
                    sys.nstates + 1,
                    sys.vardim,
                    sys.vardim
                )
            ]
            ∑u² += dot(uₜ, uₜ)
        end
        return 0.5 * Rᵤ * ∑u²
    end

    L = z -> +(
        total_time(z),
        u_smoothness_regulator(z),
        u_amplitude_regulator(z)
    )


    #
    # gradient of min time objective
    #

    ∇L = (Z::AbstractVector) -> begin

        ∇ = zeros(typeof(Z[1]), sys.vardim * T + T - 1)

        ∇[end - (T - 1) + 1:end] .= 1.0

        # u amplitude regulator gradient

        for t = 1:T-1
            uₜ_slice = slice(
                t,
                sys.nstates + 1,
                sys.vardim,
                sys.vardim
            )
            ∇[uₜ_slice] = Rᵤ * Z[uₜ_slice]
        end


        # u smoothness regulator gradient

        for t = 1:T-2

            uₜ_slice = slice(
                t,
                sys.nstates + 1,
                sys.vardim,
                sys.vardim
            )

            uₜ₊₁_slice = slice(
                t + 1,
                sys.nstates + 1,
                sys.vardim,
                sys.vardim
            )

            uₜ₊₁ = Z[uₜ₊₁_slice]

            uₜ = Z[uₜ_slice]

            Δu = uₜ₊₁ - uₜ

            ∇[uₜ_slice] += -Rₛ * Δu

            ∇[uₜ₊₁_slice] += Rₛ * Δu
        end

        return ∇
    end


    #
    # Hessian of min time objective
    #

    ∇²L = nothing
    ∇²L_structure = nothing

    if eval_hessian

        ∇²L_structure = []


        # u amplitude regulator Hessian structure

        for t = 1:T-1

            uₜ_slice = slice(
                t,
                sys.nstates + 1,
                sys.vardim,
                sys.vardim
            )

            append!(
                ∇²L_structure,
                collect(zip(uₜ_slice, uₜ_slice))
            )
        end


        # u smoothness regulator Hessian main diagonal structure

        for t = 1:T-1

            uₜ_slice = slice(
                t,
                sys.nstates + 1,
                sys.vardim,
                sys.vardim
            )

            # main diagonal (2 if t ≂̸ 1 or T-1) * Rₛ I
            # components: ∂²uₜSₜ
            append!(
                ∇²L_structure,
                collect(
                    zip(
                        uₜ_slice,
                        uₜ_slice
                    )
                )
            )
        end


        # u smoothness regulator Hessian off diagonal structure

        for t = 1:T-2

            uₜ_slice = slice(
                t,
                sys.nstates + 1,
                sys.vardim,
                sys.vardim
            )

            uₜ₊₁_slice = slice(
                t + 1,
                sys.nstates + 1,
                sys.vardim,
                sys.vardim
            )


            # off diagonal -Rₛ I components: ∂uₜ₊₁∂uₜSₜ

            append!(
                ∇²L_structure,
                collect(
                    zip(
                        uₜ_slice,
                        uₜ₊₁_slice
                    )
                )
            )
        end


        ∇²L = Z::AbstractVector -> begin

            H = []


            # u amplitude regulator Hessian values

            for t = 1:T-1
                append!(H, Rᵤ * ones(sys.ncontrols))
            end


            # u smoothness regulator Hessian main diagonal values

            append!(H, Rₛ * ones(sys.ncontrols))

            for t = 2:T-2
                append!(H, 2 * Rₛ * ones(sys.ncontrols))
            end

            append!(H, Rₛ * ones(sys.ncontrols))


            # u smoothness regulator Hessian off diagonal values

            for t = 1:T-2
                append!(H, -Rₛ * ones(sys.ncontrols))
            end

            return H
        end
    end

    return MinTimeObjective(L, ∇L, ∇²L, ∇²L_structure)
end




end
