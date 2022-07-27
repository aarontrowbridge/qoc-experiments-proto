module Losses

export QuantumStateLoss
export QuantumStateLossGradient
export QuantumStateLossHessian

export structure

export geodesic_loss
export real_loss
export amplitude_loss
export quaternionic_loss

using ..Utils
using ..QuantumLogic
using ..QubitSystems

using LinearAlgebra
using SparseArrays
using Symbolics

#
# loss functions
#


# TODO: renormalize vectors in place of abs
#       ⋅ penalize cost to remain near unit norm
#       ⋅ Σ α * (1 - ψ̃'ψ̃), α = 1e-3


struct QuantumStateLoss
    ls::Vector{Function}
    isodim::Int

    function QuantumStateLoss(sys::AbstractQubitSystem; loss=amplitude_loss)
        ls = [ψ̃ -> loss(ψ̃, sys.ψ̃goal[slice(i, sys.isodim)]) for i = 1:sys.nqstates]
        return new(ls, sys.isodim)
    end
end

function (qloss::QuantumStateLoss)(ψ̃::AbstractVector)
    loss = 0.0
    for (i, lⁱ) in enumerate(qloss.ls)
        loss += lⁱ(ψ̃[slice(i, qloss.isodim)])
    end
    return loss
end

struct QuantumStateLossGradient
    ∇ls::Vector{Function}
    isodim::Int

    function QuantumStateLossGradient(
        loss::QuantumStateLoss;
        simplify=true
    )
        Symbolics.@variables ψ̃[1:loss.isodim]
        ψ̃ = collect(ψ̃)
        ∇ls_symbs = [Symbolics.gradient(l(ψ̃), ψ̃; simplify=simplify) for l in loss.ls]
        ∇ls_exprs = [Symbolics.build_function(∇l, ψ̃) for ∇l in ∇ls_symbs]
        ∇ls = [eval(∇l_expr[1]) for ∇l_expr in ∇ls_exprs]
        return new(∇ls, loss.isodim)
    end
end

function (∇l::QuantumStateLossGradient)(ψ̃::AbstractVector)
    ∇ = similar(ψ̃)
    for (i, ∇lⁱ) in enumerate(∇l.∇ls)
        ψ̃ⁱ_slice = slice(i, ∇l.isodim)
        ∇[ψ̃ⁱ_slice] = ∇lⁱ(ψ̃[ψ̃ⁱ_slice])
    end
    return ∇
end

struct QuantumStateLossHessian
    ∇²ls::Vector{Function}
    ∇²l_structures::Vector{Vector{Tuple{Int, Int}}}
    isodim::Int

    function QuantumStateLossHessian(
        loss::QuantumStateLoss;
        simplify=true
    )

        Symbolics.@variables ψ̃[1:loss.isodim]
        ψ̃ = collect(ψ̃)

        ∇²l_symbs = [
            Symbolics.sparsehessian(
                l(ψ̃),
                ψ̃;
                simplify=simplify
            ) for l in loss.ls
        ]

        ∇²l_structures = []

        for ∇²l_symb in ∇²l_symbs
            K, J, _ = findnz(∇²l_symb)

            KJ = collect(zip(K, J))

            filter!(((k, j),) -> k ≤ j, KJ)

            push!(∇²l_structures, KJ)
        end

        ∇²l_exprs = [
            Symbolics.build_function(∇²l_symb, ψ̃)
                for ∇²l_symb in ∇²l_symbs
        ]

        ∇²ls = [eval(∇²l_expr[1]) for ∇²l_expr in ∇²l_exprs]

        return new(∇²ls, ∇²l_structures, loss.isodim)
    end
end

function structure(
    H::QuantumStateLossHessian,
    T::Int,
    vardim::Int
)
    H_structure = []
    T_offset = index(T, 0, vardim)
    for (i, KJⁱ) in enumerate(H.∇²l_structures)
        i_offset = index(i, 0, H.isodim)
        for kj in KJⁱ
            push!(H_structure, (T_offset + i_offset) .+ kj)
        end
    end
    return H_structure
end

function (H::QuantumStateLossHessian)(ψ̃::AbstractVector)
    Hs = []
    for (i, ∇²lⁱ) in enumerate(H.∇²ls)
        for (k, j) in H.∇²l_structures[i]
            ψ̃ⁱ = ψ̃[slice(i, H.isodim)]
            Hⁱᵏʲ = ∇²lⁱ(ψ̃ⁱ)[k, j]
            append!(Hs, Hⁱᵏʲ)
        end
    end
    return Hs
end

# loss functions

function geodesic_loss(ψ̃, ψ̃goal)
    ψ = iso_to_ket(ψ̃)
    ψgoal = iso_to_ket(ψ̃goal)
    amp = ψ'ψgoal
    return min(abs(1 - amp), abs(1 + amp))
end

function real_loss(ψ̃, ψ̃goal)
    ψ = iso_to_ket(ψ̃)
    ψgoal = iso_to_ket(ψ̃goal)
    amp = ψ'ψgoal
    return min(abs(1 - real(amp)), abs(1 + real(amp)))
end

function amplitude_loss(ψ̃, ψ̃goal)
    ψ = iso_to_ket(ψ̃)
    ψgoal = iso_to_ket(ψ̃goal)
    amp = ψ'ψgoal
    # return abs(1 - abs(real(amp)) + abs(imag(amp)))
    return abs(1 - abs2(amp))
end

function quaternionic_loss(ψ̃, ψ̃goal)
    return min(
        abs(1 - dot(ψ̃, ψ̃goal)),
        abs(1 + dot(ψ̃, ψ̃goal))
    )
end

end
