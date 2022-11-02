module Problems

export AbstractProblem
export FixedTimeProblem
export QuantumControlProblem

export initialize_trajectory!
export update_traj_data!
export get_traj_data
export get_variables

using ..Utils
using ..QuantumSystems
using ..Trajectories
using ..Evaluators
using ..IpoptOptions
using ..Constraints
using ..Dynamics
using ..Objectives

using Libdl
using Ipopt
using MathOptInterface
const MOI = MathOptInterface


#
#
# fixed time problems
#
#

abstract type AbstractProblem end

abstract type FixedTimeProblem <: AbstractProblem end

struct QuantumControlProblem <: FixedTimeProblem
    system::AbstractSystem
    variables::Vector{MOI.VariableIndex}
    optimizer::Ipopt.Optimizer
    objective_terms::Vector{Dict}
    constraints::Vector{AbstractConstraint}
    trajectory::Trajectory
    params::Dict
end

#
# QuantumControlProblem constructors
#

function QuantumControlProblem(
    system::AbstractSystem;
    T=100,
    Δt=0.01,
    integrator=:FourthOrderPade,
    cost=:infidelity_cost,
    Q=200.0,
    R=0.1,
    eval_hessian=true,
    pin_first_qstate=false,
    options=Options(),
    constraints::Vector{AbstractConstraint}=AbstractConstraint[],
    additional_objective=nothing,
    L1_regularized_states::Vector{Int}=Int[],
    α=fill(10.0, length(L1_regularized_states)),

    # keyword args below are for initializing the trajactory
    linearly_interpolate = true,
    σ = 0.1,
    init_traj=Trajectory(
        system,
        T,
        Δt;
        linearly_interpolate=linearly_interpolate,
        σ=σ
    ),
    kwargs...
)
    optimizer = Ipopt.Optimizer()

    set!(optimizer, options)

    n_dynamics_constraints = system.nstates * (T - 1)
    n_prob_variables = system.vardim * T

    n_variables = n_prob_variables

    params = Dict(
        :integrator => integrator,
        :cost => cost,
        :Q => Q,
        :R => R,
        :eval_hessian => eval_hessian,
        :pin_first_qstate => pin_first_qstate,
        :options => options,
        :L1_regularized_states => L1_regularized_states,
        :α => α,
        :n_prob_variables => n_prob_variables,
        :n_variables => n_variables,
        :n_dynamics_constraints => n_dynamics_constraints,
        :constraints => constraints,
    )

    quantum_obj = QuantumObjective(
        system=system,
        cost_fn=cost,
        T=T,
        Q=Q,
        eval_hessian=eval_hessian
    )

    u_regularizer = QuadraticRegularizer(
        indices=system.nstates .+ (1:system.ncontrols),
        vardim=system.vardim,
        times=1:T-1,
        R=R,
        eval_hessian=eval_hessian
    )

    objective =
        quantum_obj + u_regularizer + additional_objective

    if !isempty(L1_regularized_states)

        n_slack_variables = 2 * length(L1_regularized_states) * T

        x_indices = foldr(
            vcat,
            [
                slice(t, L1_regularized_states, system.vardim)
                    for t = 1:T
            ]
        )

        s1_indices = n_prob_variables .+ (1:n_slack_variables÷2)

        s2_indices = n_prob_variables .+
            (n_slack_variables÷2 + 1:n_slack_variables)

        params[:s1_indices] = s1_indices
        params[:s2_indices] = s2_indices

        α = foldr(vcat, [α for t = 1:T])

        L1_regularizer = L1SlackRegularizer(
            s1_indices=s1_indices,
            s2_indices=s2_indices,
            α=α,
            eval_hessian=eval_hessian
        )

        objective += L1_regularizer

        L1_slack_con = L1SlackConstraint(
            s1_indices,
            s2_indices,
            x_indices;
            name="L1 slack variable constraint"
        )

        n_variables += n_slack_variables
        params[:n_variables] = n_variables
        params[:n_slack_variables] = n_slack_variables

        constraints = vcat(constraints, L1_slack_con)
    end

    dynamics = QuantumDynamics(
        system,
        integrator,
        T,
        Δt;
        eval_hessian=eval_hessian
    )

    evaluator = PicoEvaluator(
        objective,
        dynamics,
        eval_hessian
    )

    prob_constraints = problem_constraints(
        system,
        T;
        pin_first_qstate=pin_first_qstate
    )

    cons = vcat(prob_constraints, constraints)

    variables = initialize_optimizer!(
        optimizer,
        evaluator,
        cons,
        n_dynamics_constraints,
        n_variables
    )

    return QuantumControlProblem(
        system,
        variables,
        optimizer,
        objective.terms,
        cons,
        init_traj,
        params,
    )
end

function QuantumControlProblem(
    system::AbstractSystem,
    init_traj::Trajectory;
    kwargs...
)
    return QuantumControlProblem(
        system;
        T=init_traj.T,
        Δt=init_traj.Δt,
        init_traj=init_traj,
        kwargs...
    )
end


#
# QuantumControlProblem methods
#

function initialize_optimizer!(
    optimizer::Ipopt.Optimizer,
    evaluator::PicoEvaluator,
    constraints::Vector{AbstractConstraint},
    n_dynamics_constraints::Int,
    n_variables::Int
)
    dynamics_cons = fill(
        MOI.NLPBoundsPair(0.0, 0.0),
        n_dynamics_constraints
    )
    block_data = MOI.NLPBlockData(dynamics_cons, evaluator, true)
    MOI.set(optimizer, MOI.NLPBlock(), block_data)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    variables = MOI.add_variables(optimizer, n_variables)
    constrain!(optimizer, variables, constraints)
    return variables
end


function initialize_trajectory!(
    prob::QuantumControlProblem,
    traj::Trajectory
)
    for (t, x, u) in zip(1:traj.T, traj.states, traj.actions)
        MOI.set(
            prob.optimizer,
            MOI.VariablePrimalStart(),
            prob.variables[slice(t, prob.system.vardim)],
            [x; u]
        )
    end
end

initialize_trajectory!(prob::QuantumControlProblem) =
    initialize_trajectory!(prob, prob.trajectory)

function get_variables(prob::QuantumControlProblem)
    return MOI.get(
        prob.optimizer,
        MOI.VariablePrimal(),
        prob.variables
    )
end

@views function update_traj_data!(prob::QuantumControlProblem)
    Z = get_variables(prob)

    xs = []

    for t = 1:prob.trajectory.T
        xₜ = Z[slice(t, prob.system.nstates, prob.system.vardim)]
        push!(xs, xₜ)
    end

    us = []

    for t = 1:prob.trajectory.T
        uₜ = Z[
            slice(
                t,
                prob.system.nstates + 1,
                prob.system.vardim,
                prob.system.vardim
            )
        ]
        push!(us, uₜ)
    end

    prob.trajectory.states .= xs
    prob.trajectory.actions .= us
end

end
