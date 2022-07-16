module Evaluators

export AbstractPICOEvaluator
export QubitEvaluator
export MinTimeEvaluator

using ..QubitSystems
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
    system::AbstractQubitSystem,
    integrator::Function,
    loss::Function,
    eval_hessian::Bool,
    T::Int,
    Δt::Float64,
    Q::Float64,
    Qf::Float64,
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
        loss,
        T,
        Q,
        Qf,
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
    system::AbstractQubitSystem,
    integrator::Function,
    T::Int,
    Rᵤ::Float64,
    Rₛ::Float64,
    eval_hessian::Bool
)

    dynamics = SystemDynamics(
        system,
        integrator,
        T,
        eval_hessian
    )

    objective = MinTimeObjective(
        T,
        system.nstates + 1,
        Rᵤ,
        Rₛ
    )

    return MinTimeEvaluator(
        dynamics,
        objective,
        eval_hessian
    )
end




end
