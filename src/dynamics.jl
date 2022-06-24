module Dynamics

export implicit_midpoint
export pade_schroedinger

export amplitude_loss
export real_loss
export complex_loss
export quaternionic_loss

using ..QuantumLogic

using LinearAlgebra

#
# integrators
#

# general implicit midpoint method
function implicit_midpoint(ψ̃ₜ₊₁, ψ̃ₜ, aₜ, Δt, G_drift, G_drive)
    G = G_drift + aₜ * G_drive
    return ψ̃ₜ₊₁ - (ψ̃ₜ + Δt * G * (ψ̃ₜ₊₁ + ψ̃ₜ) / 2)
end

# analytic solution of midpoint equation for schroedinger dynamics
function pade_schroedinger(ψ̃ₜ₊₁, ψ̃ₜ, aₜ, Δt, G_drift, G_drive)
    G = G_drift + aₜ * G_drive
    Id = I(size(G)[1])
    return ψ̃ₜ₊₁ - (inv(Id - Δt / 2 * G) * (Id + Δt / 2 * G) * ψ̃ₜ)
end

#
# loss functions
#

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
    return (1 - real(amp) + imag(amp))^2
end

function quaternionic_loss(ψ̃, ψ̃f)
    return min(abs(1 - dot(ψ̃, ψ̃f)), abs(1 + dot(ψ̃, ψ̃f)))
end

end
