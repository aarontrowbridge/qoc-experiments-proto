module NLMOI

export QubitProblem
export solve!

using ..QubitSystems
using ..Dynamics

using SparseArrays
using LinearAlgebra

import MathOptInterface as MOI
import Ipopt
import Symbolics

struct TrajectoryData
    states::Vector{Vector{Float64}}
    actions::Vector{Vector{Float64}}

    function TrajectoryData(system::AbstractQubitSystem{N}) where N
        state_dim = system.nqstates * system.isodim + system.control_order + 1
        action_dim = 1
        states = [randn(state_dim) for _ in 1:system.T]
        actions = [[randn(action_dim) for _ in 1:system.T-1]..., [0.0]]
        return new(states, actions)
    end
end

struct SystemDynamics
    f::Function
    ∇f::Function
    ∇f_structure::Vector{Tuple{Int, Int}}
    ∇²f::Function
    ∇²f_structure::Vector{Tuple{Int, Int}}

    function SystemDynamics(
        system::AbstractQubitSystem{N};
        eval_hessian=true
    ) where N

        xdim = system.nstates - 1

        function f(y)
            xus = [y[(1+(i-1)*system.nstates):(i*system.nstates)] for i in 1:system.T]
            xs = [xu[1:end-1] for xu in xus]
            u = [xu[end] for xu in xus]
            δxs = zeros(typeof(y[1]), xdim * (system.T - 1))
            for t = 1:system.T-1
                δxₜ₊₁ = dynamics(system, xs[t+1], xs[t], u[t])
                δxs[1+(t-1)*xdim:t*xdim] = δxₜ₊₁
            end
            return δxs
        end

        Symbolics.@variables y[1:system.nvars]

        y = collect(y)

        ∇f_symb = Symbolics.sparsejacobian(f(y), y)
        I, J, _ = findnz(∇f_symb)
        ∇f_structure = collect(zip(I, J))
        ∇f_expr = Symbolics.build_function(∇f_symb, y)
        ∇f = eval(∇f_expr[1])

        if eval_hessian
            Symbolics.@variables μ[1:xdim * (system.T - 1)]
            μ = collect(μ)
            ∇²f_symb = Symbolics.sparsehessian(dot(μ, f(y)), [y; μ])
            I, J, _ = findnz(∇²f_symb)
            ∇²f_structure = collect(zip(I, J))
            ∇²f_expr = Symbolics.build_function(∇²f_symb, y, μ)
            ∇²f = eval(∇²f_expr[1])
        else
            ∇²f = (_, _) -> nothing
            ∇²f_structure = nothing
        end

        return new(f, ∇f, ∇f_structure, ∇²f, ∇²f_structure)
    end
end

struct SystemObjective
    L::Function
    ∇L::Function
    ∇²L::Function
    ∇²L_structure::Vector{Tuple{Int,Int}}

    function SystemObjective(
        system::AbstractQubitSystem{N};
        eval_hessian=true,
        Q=0.1,
        Qf=100.0,
        R=0.5
    ) where N

        function L(y)
            xus = [y[(1+(i-1)*system.nstates):(i*system.nstates)] for i in 1:system.T]
            xs = [xu[1:end-1] for xu in xus]
            u = [xu[end] for xu in xus]
            return objective(system, xs, u, Q, Qf, R)
        end

        Symbolics.@variables y[1:system.nvars]

        y = collect(y)

        # ∇L_expr = Symbolics.sparsejacobian(L, y)
        # I, J, _ = findnz(∇L_expr)
        # ∇L_structure = collect(zip(I, J))
        # ∇L = eval(∇L_expr[1])

        ∇L_symb = Symbolics.gradient(L(y), y)
        ∇L_expr = Symbolics.build_function(∇L_symb, y)
        ∇L = eval(∇L_expr[1])

        if eval_hessian
            ∇²L_symb = Symbolics.sparsehessian(L(y), y)
            I, J, _ = findnz(∇²L_symb)
            ∇²L_structure = collect(zip(I, J))
            ∇²L_expr = Symbolics.build_function(∇²L_symb, y)
            ∇²L = eval(∇²L_expr[1])
        else
            ∇²L = (_) -> nothing
        end

        return new(L, ∇L, ∇²L, ∇²L_structure)
    end
end

# worrying about more constraints later

# struct SystemConstraints
#     gs::Vector{Function}
#     ∇gs::Vector{Function}
#     ∇²gs::Vector{Function}
# end


struct QubitEvaluator <: MOI.AbstractNLPEvaluator
    traj::TrajectoryData
    system::AbstractQubitSystem{N} where N
    dynamics::SystemDynamics
    objective::SystemObjective
    eval_hessian::Bool

    function QubitEvaluator(
        system::AbstractQubitSystem{N},
        traj::TrajectoryData;
        eval_hessian=true,
        Q=0.1,
        Qf=100.0,
        R=0.5
    ) where N

        dynamics = SystemDynamics(
            system;
            eval_hessian=eval_hessian
        )

        objective = SystemObjective(
            system;
            eval_hessian=eval_hessian,
            Q=Q,
            Qf=Qf,
            R=R
        )

        return new(
            traj,
            system,
            dynamics,
            objective,
            eval_hessian
        )
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

function MOI.eval_objective(evaluator::QubitEvaluator, y)
    evaluator.objective.L(y)
end

function MOI.eval_objective_gradient(evaluator::QubitEvaluator, ∇, y)
    ∇ .= evaluator.objective.∇L(y)
end

function MOI.eval_constraint(evaluator::QubitEvaluator, g, y)
    g .= evaluator.dynamics.f(y)
end

function MOI.eval_constraint_jacobian(evaluator::QubitEvaluator, J, y)
    ∇f = evaluator.dynamics.∇f(y)
    for (k, (i, j)) in enumerate(MOI.jacobian_structure(evaluator))
        J[k] = ∇f[i, j]
    end
end

function MOI.eval_hessian_lagrangian(evaluator::QubitEvaluator, H, y, σ, μ)
    ∇²L = σ * evaluator.objective.∇²L(y)
    for (k, (i, j)) in enumerate(evaluator.objective.∇²L_structure)
        H[k] = ∇²L[i, j]
    end
    ∇²f = evaluator.dynamics.∇²f(y, μ)
    for (k, (i, j)) in enumerate(evaluator.dynamics.∇²f_structure)
        H[length(evaluator.objective.∇²L_structure) + k] = ∇²f[i, j]
    end
end

function MOI.jacobian_structure(evaluator::QubitEvaluator)
    return evaluator.dynamics.∇f_structure
end

function MOI.hessian_lagrangian_structure(evaluator::QubitEvaluator)
    return vcat(evaluator.objective.∇²L_structure, evaluator.dynamics.∇²f_structure)
end

struct QubitProblem
    variables::Vector{MOI.VariableIndex}
    blockdata::MOI.NLPBlockData
    optimizer::Ipopt.Optimizer

    function QubitProblem(
        system::AbstractQubitSystem{N};
        eval_hessian=true,
        Q=0.1,
        Qf=100.0,
        R=0.5
    ) where N

        optimizer = Ipopt.Optimizer()

        variables = MOI.add_variables(optimizer, system.nstates * system.T)

        traj = TrajectoryData(system)

        for (t, x, u) in zip(1:system.T, traj.states, traj.actions)
            MOI.set(
                optimizer,
                MOI.VariablePrimalStart(),
                variables[(1:system.nstates) .+ (t-1)*system.nstates],
                [x; u]
            )
        end

        conbounds = [MOI.NLPBoundsPair(-0.0, 0.0) for _ = 1:(system.nstates-1)*(system.T-1)]

        evaluator = QubitEvaluator(
            system,
            traj;
            eval_hessian=eval_hessian,
            Q=Q, Qf=Qf, R=R
        )

        blockdata = MOI.NLPBlockData(conbounds, evaluator, true)

        MOI.set(optimizer, MOI.NLPBlock(), blockdata)
        MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

        return new(variables, blockdata, optimizer)
    end
end

function solve!(prob::QubitProblem)
    MOI.optimize!(prob.optimizer)
    loss = MOI.get(prob.optimizer, MOI.ObjectiveValue())
    term_status = MOI.get(prob.optimizer, MOI.TerminationStatus())
    y = MOI.get(prob.optimizer, MOI.VariablePrimal(), prob.vars)
    @show loss, term_status
    @show y
end

end
