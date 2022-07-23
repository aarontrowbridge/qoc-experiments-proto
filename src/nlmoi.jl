module NLMOI

using ..Evaluators

using MathOptInterface
const MOI = MathOptInterface

MOI.initialize(::AbstractPICOEvaluator, features) = nothing

function MOI.features_available(evaluator::AbstractPICOEvaluator)
    if evaluator.eval_hessian
        return [:Grad, :Jac, :Hess]
    else
        return [:Grad, :Jac]
    end
end

# objective and gradient

function MOI.eval_objective(evaluator::AbstractPICOEvaluator, z)
    return evaluator.objective.L(z)
end

function MOI.eval_objective_gradient(evaluator::AbstractPICOEvaluator, ∇, z)
    ∇ .= evaluator.objective.∇L(z)
    return nothing
end

# constraints and Jacobian

function MOI.eval_constraint(evaluator::AbstractPICOEvaluator, g, z)
    g .= evaluator.dynamics.F(z)
    return nothing
end

function MOI.eval_constraint_jacobian(evaluator::AbstractPICOEvaluator, J, z)
    ∇F = evaluator.dynamics.∇F(z)
    for (k, (i, j)) in enumerate(evaluator.dynamics.∇F_structure)
        J[k] = ∇F[i, j]
    end
    return nothing
end

function MOI.jacobian_structure(evaluator::AbstractPICOEvaluator)
    return evaluator.dynamics.∇f_structure
end

# Hessian of the Lagrangian

function MOI.hessian_lagrangian_structure(evaluator::AbstractPICOEvaluator)
    structure = vcat(evaluator.objective.∇²L_structure, evaluator.dynamics.∇²f_structure)
    return structure
end

function MOI.eval_hessian_lagrangian(evaluator::AbstractPICOEvaluator, H, Z, σ, μ)
    σ∇²L = σ * evaluator.objective.∇²L(Z)
    for (k, (i, j)) in enumerate(evaluator.objective.∇²L_structure)
        H[k] = σ∇²L[i, j]
    end
    μ∇²F = evaluator.dynamics.∇²f(Z, μ)
    for (k, (i, j)) in enumerate(evaluator.dynamics.∇²F_structure)
        H[length(evaluator.objective.∇²L_structure) + k] = μ∇²F[i, j]
    end
    return nothing
end

end
