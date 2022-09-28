"""
wigner.jl - functions for doing wigner function measurement
"""

#Single Qubit

using LinearAlgebra
using Random
using Distributions
import Pico: annihilate, create

include("exponential.jl")


function D(α::Union{ComplexF64, Float64}, mode_levels::Int)
    exp(α*create(mode_levels) - α' * annihilate(mode_levels))#[1:mode_levels, 1:mode_levels]
end

function phase_op(θ::Float64, mode_levels::Int; ϕ = 0.)
    Diagonal(cos.(ϕ .+ θ .* (0:mode_levels - 1)))
end

function W(α::Union{ComplexF64, Float64}, θ::Float64, mode_levels::Int; ϕ = 0.)
    2/π * D(α, mode_levels) * phase_op(θ, mode_levels, ϕ=ϕ) * D(-α, mode_levels)
end
