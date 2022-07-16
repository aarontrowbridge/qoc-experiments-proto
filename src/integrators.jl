module Integrators

export implicit_midpoint
export pade_schroedinger

using LinearAlgebra

#
# integrators
#

# TODO: add exponential and higher order pade integrators

# analytic solution of midpoint equation for schroedinger dynamics
function pade_schroedinger(
    ψ̃ₜ₊₁,
    ψ̃ₜ,
    aₜ,
    Δt::Real,
    G_drift::Matrix,
    G_drive::Matrix
)
    G = G_drift + aₜ * G_drive
    Id = I(size(G, 1))
    return (Id - Δt / 2 * G) * ψ̃ₜ₊₁ - (Id + Δt / 2 * G) * ψ̃ₜ
end

function pade_schroedinger(
    ψ̃ₜ₊₁,
    ψ̃ₜ,
    aₜs,
    Δt::Real,
    G_drift::Matrix,
    G_drives::Vector{Matrix{C}} where C <: Real
)
    G = G_drift + sum([aₜ * G_drive for (aₜ, G_drive) in zip(aₜs, G_drives)])
    Id = I(size(G, 1))
    return (Id - Δt / 2 * G) * ψ̃ₜ₊₁ - (Id + Δt / 2 * G) * ψ̃ₜ
end

end
