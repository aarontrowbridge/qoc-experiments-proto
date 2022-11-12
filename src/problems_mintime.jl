module MinTimeProblems

export MinTimeProblem
export MinTimeQuantumControlProblem

using ..Utils
using ..IpoptOptions
using ..QuantumSystems
using ..Objectives
using ..Dynamics
using ..Evaluators
using ..Constraints
using ..Trajectories
using ..Problems

using Ipopt
using MathOptInterface
const MOI = MathOptInterface


#
#
# minimum time problems
#
#

abstract type MinTimeProblem end

struct MinTimeQuantumControlProblem <: MinTimeProblem
    subprob::AbstractProblem
    optimizer::Ipopt.Optimizer
    variables::Vector{MOI.VariableIndex}
    objective_terms::Vector{Dict}
    constraints::Vector{AbstractConstraint}
    params::Dict
end



function MinTimeQuantumControlProblem(
    system::AbstractSystem;

    # time params
    T=100,
    Δt=0.01,

    # mintime prob params
    Rᵤ=0.001,
    Rₛ=0.001,
    Δt_lbound=0.1 * Δt,
    Δt_ubound=Δt,
    mintime_options=Options(),
    mintime_integrator=:FourthOrderPade,
    mintime_additional_objective=nothing,
    mintime_eval_hessian=true,

    # mintime prob  constraints
    mintime_constraints=AbstractConstraint[],

    # keyword args for initial trajectory
    σ = 0.1,
    linearly_interpolate = true,
    init_traj=Trajectory(
        system,
        T,
        Δt;
        linearly_interpolate = linearly_interpolate,
        σ = σ
    ),

    # keyword args for subprob
    kwargs...
)
    params = Dict(
        :Rᵤ => Rᵤ,
        :Rₛ => Rₛ,
        :Δt_lbound => Δt_lbound,
        :Δt_ubound => Δt_ubound,
        :mintime_options => mintime_options,
        :mintime_integrator => mintime_integrator,
        :mintime_eval_hessian => mintime_eval_hessian,
    )

    optimizer = Ipopt.Optimizer()

    set!(optimizer, mintime_options)

    subprob = QuantumControlProblem(system, init_traj; kwargs...)

    u_regularizer = QuadraticRegularizer(
        indices=system.nstates .+ (1:system.ncontrols),
        vardim=system.vardim,
        times=1:T-1,
        R=Rᵤ,
        eval_hessian=mintime_eval_hessian,
    )

    u_smoothness_regularizer = QuadraticSmoothnessRegularizer(
        indices=system.nstates .+ (1:system.ncontrols),
        vardim=system.vardim,
        times=1:T-1,
        R=Rₛ,
        eval_hessian=mintime_eval_hessian,
    )

    mintime_objective = MinTimeObjective(
        T=T,
        eval_hessian=mintime_eval_hessian
    )

    objective =
        u_regularizer +
        u_smoothness_regularizer +
        mintime_objective +
        mintime_additional_objective

    # Z̄ = [Z; Δt]
    Z_indices = 1:subprob.params[:n_variables]
    Δt_indices = subprob.params[:n_variables] .+ (1:T-1)

    params[:Z_indices] = Z_indices
    params[:Δt_indices] = Δt_indices

    dynamics = MinTimeQuantumDynamics(
        system,
        mintime_integrator,
        Z_indices,
        Δt_indices,
        T;
        eval_hessian=mintime_eval_hessian
    )

    evaluator = PicoEvaluator(
        objective,
        dynamics,
        mintime_eval_hessian
    )

    Δt_con = TimeStepBoundsConstraint(
        (Δt_lbound, Δt_ubound),
        T;
        name="time step bounds constraint"
    )

    cons = vcat(
        subprob.constraints,
        mintime_constraints,
        Δt_con
    )

    mintime_n_dynamics_constraints =
        subprob.params[:n_dynamics_constraints]

    mintime_n_variables =
        subprob.params[:n_variables] + T - 1

    params[:mintime_n_dynamics_constraints] =
        mintime_n_dynamics_constraints

    params[:mintime_n_variables] =
        mintime_n_variables


    variables = Problems.initialize_optimizer!(
        optimizer,
        evaluator,
        cons,
        mintime_n_dynamics_constraints,
        mintime_n_variables
    )

    return MinTimeQuantumControlProblem(
        subprob,
        optimizer,
        variables,
        objective.terms,
        cons,
        params
    )
end

function Problems.initialize_trajectory!(
    prob::MinTimeProblem,
    traj::Trajectory
)
    for (t, x, u) in zip(1:traj.T, traj.states, traj.actions)
        MOI.set(
            prob.optimizer,
            MOI.VariablePrimalStart(),
            prob.variables[slice(t, prob.subprob.system.vardim)],
            [x; u]
        )
    end
    MOI.set(
        prob.optimizer,
        MOI.VariablePrimalStart(),
        prob.variables[(end - (traj.T - 1) + 1):end],
        fill(traj.Δt, traj.T - 1)
    )
end

function Problems.initialize_trajectory!(prob::MinTimeProblem)
    initialize_trajectory!(prob, prob.subprob.trajectory)
end

@views function Problems.update_traj_data!(prob::MinTimeProblem)

    T       = prob.subprob.trajectory.T
    vardim  = prob.subprob.system.vardim
    nstates = prob.subprob.system.nstates

    Z = MOI.get(
        prob.optimizer,
        MOI.VariablePrimal(),
        prob.variables
    )

    xs = [Z[slice(t, nstates, vardim)] for t = 1:T]

    us = [
        Z[slice(t, nstates + 1, vardim, vardim)]
            for t = 1:T
    ]

    Δts = [0.0; Z[(end - (T - 1) + 1):end]]

    prob.subprob.trajectory.states .= xs
    prob.subprob.trajectory.actions .= us
    prob.subprob.trajectory.times .= cumsum(Δts)
end



end
