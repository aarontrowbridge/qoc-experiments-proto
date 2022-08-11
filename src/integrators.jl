module Integrators

export AbstractQuantumIntegrator

export Exponential
export SecondOrderPade
export FourthOrderPade

export Jacobian

export SecondOrderPadeJacobian
export FourthOrderPadeJacobian

export SecondOrderPadeHessian
export FourthOrderPadeHessian

export G

using ..Utils
using ..QubitSystems
using ..Utils

using LinearAlgebra


# G(a) helper function

function G(
    a::AbstractVector,
    G_drift::AbstractMatrix,
    G_drives::AbstractVector{<:AbstractMatrix}
)
    return G_drift + sum(a .* G_drives)
end


#
# integrator types
#

abstract type AbstractQuantumIntegrator end


# exponential

struct Exponential <: AbstractQuantumIntegrator
    G_drift::Matrix
    G_drives::Vector{Matrix}

    Exponential(sys::AbstractQubitSystem) =
        new(sys.G_drift, sys.G_drives)
end

function (integrator::Exponential)(
    ψ̃ₜ₊₁::AbstractVector,
    ψ̃ₜ::AbstractVector,
    aₜ::AbstractVector,
    Δt::Real,
)
    Gₜ = G(aₜ, integrator.G_drift, integrator.G_drives)
    return ψ̃ₜ₊₁ - exp(Gₜ * Δt) * ψ̃ₜ
end


# 2nd order Pade integrator

struct SecondOrderPade <: AbstractQuantumIntegrator
    G_drift::Matrix
    G_drives::Vector{Matrix}

    SecondOrderPade(sys::AbstractQubitSystem) =
        new(sys.G_drift, sys.G_drives)
end

function (integrator::SecondOrderPade)(
    ψ̃ₜ₊₁::AbstractVector,
    ψ̃ₜ::AbstractVector,
    aₜ::AbstractVector,
    Δt::Real
)
    Gₜ = G(aₜ, integrator.G_drift, integrator.G_drives)
    # Id = I(size(Gₜ, 1))
    # return (Id - Δt / 2 * Gₜ) * ψ̃ₜ₊₁ -
    #        (Id + Δt / 2 * Gₜ) * ψ̃ₜ
    return ψ̃ₜ₊₁ - ψ̃ₜ - Δt / 2 * Gₜ * (ψ̃ₜ₊₁ + ψ̃ₜ)
end


# 4th order Pade integrator

struct FourthOrderPade <: AbstractQuantumIntegrator
    G_drift::Matrix
    G_drives::Vector{Matrix}

    FourthOrderPade(sys::AbstractQubitSystem) =
        new(sys.G_drift, sys.G_drives)
end

function (integrator::FourthOrderPade)(
    ψ̃ₜ₊₁::AbstractVector,
    ψ̃ₜ::AbstractVector,
    aₜ::AbstractVector,
    Δt::Real
)
    Gₜ = G(aₜ, integrator.G_drift, integrator.G_drives)
    Id = I(size(Gₜ, 1))
    # return (Id - Δt / 2 * Gₜ + Δt^2 / 9 * Gₜ^2) * ψ̃ₜ₊₁ -
    #        (Id + Δt / 2 * Gₜ + Δt^2 / 9 * Gₜ^2) * ψ̃ₜ
    return (Id + Δt^2 / 9 * Gₜ^2) * (ψ̃ₜ₊₁ - ψ̃ₜ) -
        Δt / 2 * Gₜ * (ψ̃ₜ₊₁ + ψ̃ₜ)
end


#
# Jacobians
#

function Jacobian(integrator::AbstractQuantumIntegrator)
    if isa(integrator, SecondOrderPade)
        return SecondOrderPadeJacobian(integrator)
    elseif isa(integrator, FourthOrderPade)
        return FourthOrderPadeJacobian(integrator)
    end
end


# 2nd order Pade integrator Jacobian struct

struct SecondOrderPadeJacobian
    G_drift::Matrix
    G_drives::Vector{Matrix}

    function SecondOrderPadeJacobian(integrator::AbstractQuantumIntegrator)
        return new(integrator.G_drift, integrator.G_drives)
    end
end


# Jacobian of 2nd order Pade integrator with respect to ψ̃ⁱₜ

function (J::SecondOrderPadeJacobian)(
    aₜ::AbstractVector,
    Δt::Real,
    ψ̃ⁱₜ₊₁::Bool
)
    Gₜ = G(aₜ, J.G_drift, J.G_drives)
    Id = I(size(Gₜ, 1))
    if ψ̃ⁱₜ₊₁
        return Id - Δt / 2 * Gₜ
    else
        return -(Id + Δt / 2 * Gₜ)
    end
end


# Jacobian of 2nd order Pade integrator with respect to control aʲₜ

function (J::SecondOrderPadeJacobian)(
    ψ̃ⁱₜ₊₁::AbstractVector,
    ψ̃ⁱₜ::AbstractVector,
    aₜ::AbstractVector, # not used here, but kept to match function signature to 4th order P
    Δt::Real,
    j::Int
)
    Gʲ = J.G_drives[j]
    return -Δt / 2 * Gʲ * (ψ̃ⁱₜ₊₁ + ψ̃ⁱₜ)
end


# 4th order Pade integrator Jacobian struct

struct FourthOrderPadeJacobian
    G_drift::Matrix
    G_drives::Vector{Matrix}
    G_drift_anticoms::Vector{Matrix}
    G_drive_anticoms::Symmetric

    function FourthOrderPadeJacobian(
        integrator::AbstractQuantumIntegrator
    )
        ncontrols = length(integrator.G_drives)

        drive_anticoms = fill(
            zeros(size(integrator.G_drift)),
            ncontrols,
            ncontrols
        )

        for j = 1:ncontrols
            for k = 1:j
                if k == j
                    drive_anticoms[k, k] =
                        2 * integrator.G_drives[k]^2
                else
                    drive_anticoms[k, j] = anticom(
                        integrator.G_drives[k],
                        integrator.G_drives[j]
                    )
                end
            end
        end

        drift_anticoms = [
            anticom(G_drive, integrator.G_drift)
                for G_drive in integrator.G_drives
        ]

        return new(
            integrator.G_drift,
            integrator.G_drives,
            drift_anticoms,
            Symmetric(drive_anticoms)
        )
    end
end

anticom(A::Matrix, B::Matrix) = A * B + B * A


# Jacobian of 4th order Pade integrator with respect to ψ̃ⁱₜ

function (J::FourthOrderPadeJacobian)(
    aₜ::AbstractVector,
    Δt::Real,
    ψ̃ⁱₜ₊₁::Bool
)
    Gₜ = G(aₜ, J.G_drift, J.G_drives)
    Id = I(size(Gₜ, 1))
    if ψ̃ⁱₜ₊₁
        return Id - Δt / 2 * Gₜ + Δt^2 / 9 * Gₜ^2
    else
        return -(Id + Δt / 2 * Gₜ + Δt^2 / 9 * Gₜ^2)
    end
end


# Jacobian of 2nd order Pade integrator with respect to control aʲₜ

function (J::FourthOrderPadeJacobian)(
    ψ̃ⁱₜ₊₁::AbstractVector,
    ψ̃ⁱₜ::AbstractVector,
    aₜ::AbstractVector,
    Δt::Real,
    j::Int
)
    Gʲ_anticom_Gₜ = G(aₜ, J.G_drift_anticoms[j], J.G_drive_anticoms[:, j])
    Gʲ = J.G_drives[j]
    return -Δt / 2 * Gʲ * (ψ̃ⁱₜ₊₁ + ψ̃ⁱₜ) + Δt^2 / 9 * Gʲ_anticom_Gₜ * (ψ̃ⁱₜ₊₁ - ψ̃ⁱₜ)
end


# 2nd order Pade integrator Hessian struct

struct SecondOrderPadeHessian
    G_drives::Vector{Matrix}
    isodim::Int

    function SecondOrderPadeHessian(sys::AbstractQubitSystem)
        return new(
            sys.G_drives,
            sys.isodim
        )
    end
end


# Hessian of 2nd order Pade integrator w.r.t. ψ̃ⁱₜ₍₊₁₎ and aʲₜ

@views function (H::SecondOrderPadeHessian)(
    μₜ::AbstractVector,
    Δt::Real,
    i::Int,
    j::Int,
    ψ̃ⁱₜ₊₁::Bool
)
    Gʲ = H.G_drives[j]
    μⁱₜ = μₜ[slice(i, H.isodim)]
    if ψ̃ⁱₜ₊₁
        return (μⁱₜ)' * (-Δt / 2 * Gʲ)
    else
        return -(Δt / 2 * Gʲ)' * μⁱₜ
    end
end


# 4th order Pade integrator Hessian struct

struct FourthOrderPadeHessian
    G_drives::Vector{Matrix}
    G_drive_anticoms::Symmetric
    G_drift_anticoms::Vector{Matrix}
    nqstates::Int
    isodim::Int

    function FourthOrderPadeHessian(sys::AbstractQubitSystem)
        drive_anticoms = fill(
            zeros(size(sys.G_drift)),
            sys.ncontrols,
            sys.ncontrols
        )

        for j = 1:sys.ncontrols
            for k = 1:j
                if k == j
                    drive_anticoms[k, k] = 2 * sys.G_drives[k]^2
                else
                    drive_anticoms[k, j] =
                        anticom(sys.G_drives[k], sys.G_drives[j])
                end
            end
        end

        drift_anticoms = [
            anticom(G_drive, sys.G_drift)
                for G_drive in sys.G_drives
        ]

        return new(
            sys.G_drives,
            Symmetric(drive_anticoms),
            drift_anticoms,
            sys.nqstates,
            sys.isodim
        )
    end
end


# Hessian of the 4th order Pade integrator w.r.t. aᵏₜ and aʲₜ

@views function (H::FourthOrderPadeHessian)(
    ψ̃ₜ₊₁::AbstractVector,
    ψ̃ₜ::AbstractVector,
    μₜ::AbstractVector,
    Δt::Real,
    k::Int,
    j::Int
)
    Hᵏʲₜ = 0.0

    Ĝᵏʲ = Δt^2 / 9 * H.G_drive_anticoms[k, j]

    for i = 1:H.nqstates

        ψ̃ⁱ_slice = slice(i, H.isodim)

        μⁱₜ = μₜ[ψ̃ⁱ_slice]

        ψ̃ⁱₜ = ψ̃ₜ[ψ̃ⁱ_slice]

        ψ̃ⁱₜ₊₁ = ψ̃ₜ₊₁[ψ̃ⁱ_slice]

        Hᵏʲₜ += dot(μⁱₜ, Ĝᵏʲ * (ψ̃ⁱₜ₊₁ - ψ̃ⁱₜ))
    end

    return Hᵏʲₜ
end


# Hessian of 4th order Pade integrator w.r.t. ψ̃ⁱₜ₍₊₁₎  and aʲₜ

@views function (H::FourthOrderPadeHessian)(
    aₜ::AbstractVector,
    μₜ::AbstractVector,
    Δt::Real,
    i::Int,
    j::Int,
    ψ̃ⁱₜ₊₁::Bool
)
    Ĝʲ = G(aₜ, H.G_drift_anticoms[j], H.G_drive_anticoms[:, j])
    Gʲ = H.G_drives[j]
    μⁱₜ = μₜ[slice(i, H.isodim)]
    if ψ̃ⁱₜ₊₁
        return (μⁱₜ)' * (-Δt / 2 * Gʲ + Δt^2 / 9 * Ĝʲ)
    else
        return -(Δt / 2 * Gʲ + Δt^2 / 9 * Ĝʲ)' * μⁱₜ
    end
end




end
