module Problems

export FixedTimeProblem
export MinTimeProblem
export QuantumControlProblem
export QuantumMinTimeProblem
export ProblemData

export solve!
export initialize_trajectory!
export update_traj_data!
export get_traj_data

export save_prob
export load_prob

using ..Utils
using ..QuantumSystems
using ..Trajectories
using ..NLMOI
using ..Evaluators
using ..IpoptOptions
using ..Constraints

using JLD2
using Libdl
using Ipopt
using MathOptInterface
const MOI = MathOptInterface


#
# qubit problem
#

abstract type FixedTimeProblem end

struct ProblemData
    system::AbstractQuantumSystem
    trajectory::Trajectory
    constraints::Vector{AbstractConstraint}
    params::Dict

    function ProblemData(prob::FixedTimeProblem)
        return new(
            prob.system,
            prob.trajectory,
            prob.constraints,
            prob.params
        )
    end
end

function save_prob(prob::FixedTimeProblem, path::String)
    path_parts = split(path, "/")
    dir = joinpath(path_parts[1:end-1])
    if !isdir(dir)
        mkpath(dir)
    end
    data = ProblemData(prob)
    @save path data
end

function load_prob(path::String)
    @load path data
    return QuantumControlProblem(data)
end



struct QuantumControlProblem <: FixedTimeProblem
    system::AbstractQuantumSystem
    variables::Vector{MOI.VariableIndex}
    optimizer::Ipopt.Optimizer
    trajectory::Trajectory
    constraints::Vector{AbstractConstraint}
    params::Dict
end

function QuantumControlProblem(
    system::AbstractQuantumSystem,
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

function QuantumControlProblem(data::ProblemData)
    return QuantumControlProblem(
        data.system,
        data.trajectory,
        data.constraints,
        data.params
    )
end

function QuantumControlProblem(
    sys::AbstractQuantumSystem,
    traj::Trajectory,
    cons::Vector{AbstractConstraint},
    params::Dict
)
    optimizer = Ipopt.Optimizer()

    set!(optimizer, params[:options])

    evaluator = QuantumEvaluator(
        sys,
        params[:integrator],
        params[:cost],
        params[:eval_hessian],
        traj.T,
        traj.Δt,
        params[:Q],
        params[:R]
    )

    n_dynamics_cons = sys.nstates * (traj.T - 1)
    n_variables = sys.vardim * traj.T

    variables = initialize_optimizer!(
        optimizer,
        evaluator,
        n_dynamics_cons,
        n_variables
    )

    constrain!(optimizer, variables, cons)

    return QuantumControlProblem(
        sys,
        variables,
        optimizer,
        traj,
        cons,
        params,
    )
end



function QuantumControlProblem(
    system::AbstractQuantumSystem;
    T=100,
    Δt=0.01,
    integrator=:FourthOrderPade,
    cost=:infidelity_cost,
    Q=200.0,
    R=0.1,
    eval_hessian=true,
    pin_first_qstate=true,
    options=Options(),
    constraints=AbstractConstraint[],

    # keyword args below are for initializing the trajactory
    linearly_interpolate = true,
    σ = 0.1,
    init_traj=Trajectory(
        system,
        T,
        Δt;
        linearly_interpolate=linearly_interpolate,
        σ=σ
    )
)
    params = Dict(
        :integrator => integrator,
        :cost => cost,
        :Q => Q,
        :R => R,
        :eval_hessian => eval_hessian,
        :pin_first_qstate => pin_first_qstate,
        :options => options,
    )

    optimizer = Ipopt.Optimizer()

    set!(optimizer, options)

    evaluator = QuantumEvaluator(
        system,
        integrator,
        cost,
        eval_hessian,
        T, Δt,
        Q, R
    )

    n_dynamics_cons = system.nstates * (T - 1)
    n_variables = system.vardim * T

    variables = initialize_optimizer!(
        optimizer,
        evaluator,
        n_dynamics_cons,
        n_variables
    )

    cons = problem_constraints(
        system,
        T;
        pin_first_qstate=pin_first_qstate
    )

    push!(cons, constraints...)

    constrain!(optimizer, variables, cons)

    return QuantumControlProblem(
        system,
        variables,
        optimizer,
        init_traj,
        cons,
        params,
    )
end

function initialize_optimizer!(
    optimizer::Ipopt.Optimizer,
    evaluator::AbstractPICOEvaluator,
    n_dynamics_cons::Int,
    n_variables::Int
)
    dynamics_cons = fill(
        MOI.NLPBoundsPair(0.0, 0.0),
        n_dynamics_cons
    )
    block_data = MOI.NLPBlockData(dynamics_cons, evaluator, true)
    MOI.set(optimizer, MOI.NLPBlock(), block_data)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    variables = MOI.add_variables(optimizer, n_variables)
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

initialize_trajectory!(prob) =
    initialize_trajectory!(prob, prob.trajectory)

@views function update_traj_data!(prob::QuantumControlProblem)
    Z = MOI.get(
        prob.optimizer,
        MOI.VariablePrimal(),
        prob.variables
    )

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

function solve!(
    prob::QuantumControlProblem;
    init_traj = prob.trajectory,
    save_path = nothing
)

    initialize_trajectory!(prob, init_traj)

    MOI.optimize!(prob.optimizer)

    update_traj_data!(prob)

    if ! isnothing(save_path)
        save_prob(prob, save_path)
    end
end


#
# min time problem
#

abstract type MinTimeProblem end

struct QuantumMinTimeProblem <: MinTimeProblem
    subprob::QuantumControlProblem
    evaluator::MinTimeEvaluator
    variables::Vector{MOI.VariableIndex}
    optimizer::Ipopt.Optimizer
    constraints::Vector{AbstractConstraint}
    params::Dict
end

# TODO: rewrite this constructor (hacky implementation rn)

function QuantumMinTimeProblem(
    data::ProblemData;
    Rᵤ=0.001,
    Rₛ=0.001,
    Δt_lbound=0.1 * prob.trajectory.Δt,
    Δt_ubound=prob.trajectory.Δt,
    mintime_eval_hessian=true,
    mintime_options=Options(),
    mintime_constraints=AbstractConstraint[]
)
    params = Dict(
        :Rᵤ => Rᵤ,
        :Rₛ => Rₛ,
        :Δt_lbound => Δt_lbound,
        :Δt_ubound => Δt_ubound,
        :eval_hessian => eval_hessian,
        :mintime_options => mintime_options,
    )

    optimizer = Ipopt.Optimizer()

    set!(optimizer, mintime_options)

    T = data.trajectory.T

    evaluator = MinTimeEvaluator(
        data.system,
        data.params[:integrator],
        T,
        Rᵤ,
        Rₛ,
        mintime_eval_hessian
    )

    n_dynamics_cons = data.system.nstates * (T - 1)
    n_variables = data.system.vardim * T + T - 1

    variables = initialize_optimizer!(
        optimizer,
        evaluator,
        n_dynamics_cons,
        n_variables
    )

    cons = vcat(data.constraints, mintime_constraints)

    Δt_con = TimeStepBoundsConstraint(
        (Δt_lbound, Δt_ubound),
        T;
        name="time step bounds constraint"
    )

    push!(cons, Δt_con)

    constrain!(optimizer, variables, cons)

    return QuantumMinTimeProblem(
        prob,
        evaluator,
        variables,
        optimizer,
        cons,
        params
    )
end



function QuantumMinTimeProblem(
    system::AbstractQuantumSystem;

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

    evaluator = MinTimeEvaluator(
        system,
        mintime_integrator,
        T,
        Rᵤ,
        Rₛ,
        mintime_eval_hessian
    )

    n_dynamics_cons = system.nstates * (T - 1)
    n_variables = system.vardim * T + T - 1

    variables = initialize_optimizer!(
        optimizer,
        evaluator,
        n_dynamics_cons,
        n_variables
    )

    cons = vcat(subprob.constraints, mintime_constraints)

    Δt_con = TimeStepBoundsConstraint(
        (Δt_lbound, Δt_ubound),
        T;
        name="time step bounds constraint"
    )

    push!(cons, Δt_con)

    constrain!(optimizer, variables, cons)

    return QuantumMinTimeProblem(
        subprob,
        evaluator,
        variables,
        optimizer,
        cons,
        params
    )
end

function initialize_trajectory!(
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

function initialize_trajectory!(prob::MinTimeProblem)
    initialize_trajectory!(prob, prob.subprob.trajectory)
end

@views function update_traj_data!(prob::MinTimeProblem)

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


function solve!(
    prob::MinTimeProblem;
    save_path=nothing,
    solve_subprob=true,
)
    if solve_subprob
        solve!(prob.subprob)
    end

    init_traj = prob.subprob.trajectory

    initialize_trajectory!(prob, init_traj)

    n_wfn_states = prob.subprob.system.n_wfn_states

    # constrain endpoints to match subprob solution

    if prob.subprob.params[:pin_first_qstate]
        isodim = prob.subprob.system.isodim
        ψ̃T_con! = EqualityConstraint(
            prob.subprob.trajectory.T,
            (isodim + 1):n_wfn_states,
            init_traj.states[end][(isodim + 1):n_wfn_states],
            prob.subprob.system.vardim;
            name="final qstate constraint"
        )
    else
        ψ̃T_con! = EqualityConstraint(
            prob.subprob.trajectory.T,
            1:n_wfn_states,
            init_traj.states[end][1:n_wfn_states],
            prob.subprob.system.vardim;
            name="final qstate constraint"
        )
    end

    ψ̃T_con!(prob.optimizer, prob.variables)

    MOI.optimize!(prob.optimizer)

    update_traj_data!(prob)

    if ! isnothing(save_path)
        save_prob(prob.subprob, save_path)
    end
end


# TODO: add functionality to vizualize Δt distribution


end
