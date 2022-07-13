module Integrators

export implicit_midpoint
export pade_schroedinger

using LinearAlgebra

#
# integrators
#

# TODO: add exponential and higher order pade integrators

# analytic solution of midpoint equation for schroedinger dynamics
function pade_schroedinger(ψ̃ₜ₊₁, ψ̃ₜ, aₜ, Δt, G_drift, G_drive)
    G = G_drift + aₜ * G_drive
    Id = I(size(G)[1])
    return (Id - Δt / 2 * G) * ψ̃ₜ₊₁ - (Id + Δt / 2 * G) * ψ̃ₜ
end

end
