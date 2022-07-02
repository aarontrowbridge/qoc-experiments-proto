module Evaluators

export QubitEvaluator
export MinTimeEvaluator

using ..QubitSystems
using ..Dynamics
using ..Objectives

using MathOptInterface
const MOI = MathOptInterface

import Ipopt

struct QubitEvaluator <: MOI.AbstractNLPEvaluator
    dynamics::SystemDynamics
    objective::SystemObjective
    eval_hessian::Bool
end

function QubitEvaluator(
    system::AbstractQubitSystem{N},
    integrator::Function,
    loss::Function,
    eval_hessian::Bool,
    T::Int,
    Δt::Float64,
    Q::Float64,
    Qf::Float64,
    R::Float64
) where N

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

struct MinTimeEvaluator <: MOI.AbstractNLPEvaluator
    dynamics::SystemDynamics
    objective::MinTimeObjective
    eval_hessian::Bool
end

function MinTimeEvaluator(
    system::AbstractQubitSystem{N},
    integrator::Function,
    T::Int,
    B::Float64,
    eval_hessian::Bool,
    squared_loss::Bool
) where N

    dynamics = SystemDynamics(
        system,
        integrator,
        T,
        eval_hessian
    )

    objective = MinTimeObjective(
        T,
        system.nstates + 1,
        B,
        squared_loss
    )

    return MinTimeEvaluator(
        dynamics,
        objective,
        eval_hessian
    )
end




end
