module Evaluators

export AbstractPICOEvaluator
export QuantumEvaluator
export MinTimeEvaluator

using ..QuantumSystems
using ..Integrators
using ..Dynamics
using ..Objectives

using MathOptInterface
const MOI = MathOptInterface

import Ipopt

abstract type AbstractPICOEvaluator <: MOI.AbstractNLPEvaluator end

struct QuantumEvaluator <: AbstractPICOEvaluator
    dynamics::QuantumDynamics
    objective::QuantumObjective
    eval_hessian::Bool
end

function QuantumEvaluator(
    system::AbstractQuantumSystem,
    integrator::Symbol,
    cost_fn::Symbol,
    eval_hessian::Bool,
    T::Int,
    Δt::Float64,
    Q::Float64,
    R::Float64
)

    dynamics = QuantumDynamics(
        system,
        integrator,
        T,
        Δt,
        eval_hessian
    )

    objective = QuantumObjective(
        system,
        cost_fn,
        T,
        Q,
        R,
        eval_hessian
    )

    return QuantumEvaluator(
        dynamics,
        objective,
        eval_hessian
    )
end

struct MinTimeEvaluator <: AbstractPICOEvaluator
    dynamics::QuantumDynamics
    objective::MinTimeObjective
    eval_hessian::Bool
end

function MinTimeEvaluator(
    system::AbstractQuantumSystem,
    integrator::Symbol,
    T::Int,
    Rᵤ::Float64,
    Rₛ::Float64,
    eval_hessian::Bool
)

    dynamics = MinTimeQuantumDynamics(
        system,
        integrator,
        T,
        eval_hessian
    )

    objective = MinTimeObjective(
        system,
        T,
        Rᵤ,
        Rₛ,
        eval_hessian
    )

    return MinTimeEvaluator(
        dynamics,
        objective,
        eval_hessian
    )
end




end
