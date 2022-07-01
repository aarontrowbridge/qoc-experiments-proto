module NLMOI

using MathOptInterface
const MOI = MathOptInterface

MOI.initialize(::MOI.AbstractNLPEvaluator, features) = nothing

function MOI.features_available(evaluator::MOI.AbstractNLPEvaluator)
    if evaluator.eval_hessian
        return [:Grad, :Jac, :Hess]
    else
        return [:Grad, :Jac]
    end
end

# objective and gradient

function MOI.eval_objective(evaluator::MOI.AbstractNLPEvaluator, y)
    return evaluator.objective.L(y)
end

function MOI.eval_objective_gradient(evaluator::MOI.AbstractNLPEvaluator, ∇, y)
    ∇ .= evaluator.objective.∇L(y)
    return nothing
end

# constraints and Jacobian

function MOI.eval_constraint(evaluator::MOI.AbstractNLPEvaluator, g, y)
    g .= evaluator.dynamics.f(y)
    return nothing
end

function MOI.eval_constraint_jacobian(evaluator::MOI.AbstractNLPEvaluator, J, y)
    ∇f = evaluator.dynamics.∇f(y)
    for (k, (i, j)) in enumerate(evaluator.dynamics.∇f_structure)
        J[k] = ∇f[i, j]
    end
    return nothing
end

function MOI.jacobian_structure(evaluator::MOI.AbstractNLPEvaluator)
    return evaluator.dynamics.∇f_structure
end

# Hessian of the Lagrangian

function MOI.eval_hessian_lagrangian(evaluator::MOI.AbstractNLPEvaluator, H, y, σ, μ)
    σ∇²L = σ * evaluator.objective.∇²L(y)
    for (k, (i, j)) in enumerate(evaluator.objective.∇²L_structure)
        H[k] = σ∇²L[i, j]
    end
    μ∇²f = evaluator.dynamics.∇²f(y, μ)
    for (k, (i, j)) in enumerate(evaluator.dynamics.∇²f_structure)
        H[length(evaluator.objective.∇²L_structure) + k] = μ∇²f[i, j]
    end
    return nothing
end

function MOI.hessian_lagrangian_structure(evaluator::MOI.AbstractNLPEvaluator)
    structure = vcat(evaluator.objective.∇²L_structure, evaluator.dynamics.∇²f_structure)
    return structure
end

end
