module Losses

export geodesic_loss
export real_loss
export amplitude_loss
export quaternionic_loss

using ..QuantumLogic

using LinearAlgebra

#
# loss functions
#


# TODO: renormalize vectors in place of abs
#       ⋅ penalize cost to remain near unit norm
#       ⋅ Σ α * (1 - ψ̃'ψ̃), α = 1e-3

function geodesic_loss(ψ̃, ψ̃f)
    ψ = iso_to_ket(ψ̃)
    ψf = iso_to_ket(ψ̃f)
    amp = ψ'ψf
    return min(abs(1 - amp), abs(1 + amp))
end

function real_loss(ψ̃, ψ̃f)
    ψ = iso_to_ket(ψ̃)
    ψf = iso_to_ket(ψ̃f)
    amp = ψ'ψf
    return min(abs(1 - real(amp)), abs(1 + real(amp)))
end

function amplitude_loss(ψ̃, ψ̃f)
    ψ = iso_to_ket(ψ̃)
    ψf = iso_to_ket(ψ̃f)
    amp = ψ'ψf
    # return abs(1 - abs(real(amp)) + abs(imag(amp)))
    return abs(1 - abs2(amp))
end

function quaternionic_loss(ψ̃, ψ̃f)
    return min(abs(1 - dot(ψ̃, ψ̃f)), abs(1 + dot(ψ̃, ψ̃f)))
end

end
