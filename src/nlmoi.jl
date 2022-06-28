module NLMOI

export QubitEvaluator

using ..QubitSystems
using ..Dynamics
using ..Objective

using MathOptInterface
const MOI = MathOptInterface

import Ipopt

struct QubitEvaluator <: MOI.AbstractNLPEvaluator
    dynamics::SystemDynamics
    objective::SystemObjective
    eval_hessian::Bool

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
        dynamics = SystemDynamics(system, integrator, eval_hessian, T, Δt)
        objective = SystemObjective(system, loss, eval_hessian, T, Q, Qf, R)
        return new(dynamics, objective, eval_hessian)
    end
end

MOI.initialize(::QubitEvaluator, features) = nothing

function MOI.features_available(evaluator::QubitEvaluator)
    if evaluator.eval_hessian
        return [:Grad, :Jac, :Hess]
    else
        return [:Grad, :Jac]
    end
end

# objective and gradient

function MOI.eval_objective(evaluator::QubitEvaluator, y)
    # @info "howdy from eval_objective!"
    return evaluator.objective.L(y)
end

function MOI.eval_objective_gradient(evaluator::QubitEvaluator, ∇, y)::Nothing
    ∇ .= evaluator.objective.∇L(y)
    # @info "howdy from eval_objective_gradient!"
    return nothing
end

# constraints and Jacobian

function MOI.eval_constraint(evaluator::QubitEvaluator, g, y)::Nothing
    g .= evaluator.dynamics.f(y)
    # @info "howdy from eval_constraint!"
    return nothing
end

function MOI.eval_constraint_jacobian(evaluator::QubitEvaluator, J, y)::Nothing
    ∇f = evaluator.dynamics.∇f(y)
    for (k, (i, j)) in enumerate(evaluator.dynamics.∇f_structure)
        J[k] = ∇f[i, j]
    end
    # @info "howdy from eval_constraint_jacobian!"
    return nothing
end

function MOI.jacobian_structure(evaluator::QubitEvaluator)
    # @info "howdy from jacobian_structure!"
    return evaluator.dynamics.∇f_structure
end

# Hessian of the Lagrangian

function MOI.eval_hessian_lagrangian(evaluator::QubitEvaluator, H, y, σ, μ)::Nothing
    σ∇²L = σ * evaluator.objective.∇²L(y)
    for (k, (i, j)) in enumerate(evaluator.objective.∇²L_structure)
        H[k] = σ∇²L[i, j]
    end
    μ∇²f = evaluator.dynamics.∇²f(y, μ)
    for (k, (i, j)) in enumerate(evaluator.dynamics.∇²f_structure)
        H[length(evaluator.objective.∇²L_structure) + k] = μ∇²f[i, j]
    end
    # @info "howdy from eval_hessian_lagrangian!"
    return nothing
end

function MOI.hessian_lagrangian_structure(evaluator::QubitEvaluator)
    @info "howdy from hessian_lagrangian_structure!"
    structure = vcat(evaluator.objective.∇²L_structure, evaluator.dynamics.∇²f_structure)
    @info "structure check" typeof(structure)
    return structure
end

end
