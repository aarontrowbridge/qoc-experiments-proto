module Evaluators

export AbstractPICOEvaluator
export QubitEvaluator
export MinTimeEvaluator

using ..QubitSystems
using ..Integrators
using ..Dynamics
using ..Objectives

using MathOptInterface
const MOI = MathOptInterface

import Ipopt

abstract type AbstractPICOEvaluator <: MOI.AbstractNLPEvaluator end

struct QubitEvaluator <: AbstractPICOEvaluator
    dynamics::SystemDynamics
    objective::SystemObjective
    eval_hessian::Bool
end

function QubitEvaluator(
    system::AbstractQuantumSystem,
    integrator::Symbol,
    cost_fn::Function,
    eval_hessian::Bool,
    T::Int,
    Δt::Float64,
    Q::Float64,
    R::Float64
)

    dynamics = SystemDynamics(
        system,
        integrator,
        T,
        Δt,
        eval_hessian
    )

    objective = SystemObjective(
        system,
        cost_fn,
        T,
        Q,
        R,
        eval_hessian
    )

    return QubitEvaluator(
        dynamics,
        objective,
        eval_hessian
    )
end

struct MinTimeEvaluator <: AbstractPICOEvaluator
    dynamics::SystemDynamics
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

    dynamics = MinTimeSystemDynamics(
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
