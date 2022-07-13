module Problems

export QubitProblem
export MinTimeProblem

export solve!
export initialize_trajectory!
export update_traj_data!
export get_traj_data

using ..Utils
using ..Integrators
using ..Losses
using ..QubitSystems
using ..Trajectories
using ..NLMOI
using ..Evaluators
using ..IpoptOptions

using Ipopt
using MathOptInterface
const MOI = MathOptInterface


#
# qubit problem
#

struct QubitProblem
    system::AbstractQubitSystem{N} where N
    evaluator::QubitEvaluator
    variables::Vector{MOI.VariableIndex}
    optimizer::Ipopt.Optimizer
    trajectory::TrajectoryData
    T::Int
    Δt::Float64
    vardim::Int
    total_vars::Int
    total_states::Int
    total_dynamics::Int
end

function QubitProblem(
    system::AbstractQubitSystem{N},
    T::Int;
    integrator=pade_schroedinger,
    loss=quaternionic_loss,
    Δt=0.01,
    Q=0.1,
    Qf=100.0,
    R=0.5,
    σ=1.0,
    eval_hessian=false,
    pin_first_qstate=true,
    bound_a=true,
    a_bound=1.0,
    options=Options(),
) where N

    optimizer = Ipopt.Optimizer()

    # set Ipopt optimizer options
    for name in fieldnames(typeof(options))
        optimizer.options[String(name)] = getfield(options, name)
    end

    vardim = system.nstates + 1
    total_vars = vardim * T
    total_dynamics = system.nstates * (T - 1)
    total_states = system.nstates * T

    variables = MOI.add_variables(optimizer, total_vars)

    evaluator = QubitEvaluator(
        system,
        integrator,
        loss,
        eval_hessian,
        T, Δt,
        Q, Qf, R
    )

    n_wfn_states = system.nqstates * system.isodim

    # initial quantum state constraints: ψ̃(t=1) = ψ̃1
    for i = 1:n_wfn_states
        MOI.add_constraints(
            optimizer,
            variables[i],
            MOI.EqualTo(system.ψ̃1[i])
        )
    end

    # pin first qstate to be equal to analytic solution
    if pin_first_qstate
        for i = 1:system.isodim
            MOI.add_constraints(
                optimizer,
                variables[index(T, i, vardim)],
                MOI.EqualTo(system.ψ̃goal[i])
            )
        end
    end

    # initial a(t = 1) constraints: ∫a, a, da = 0
    for i = 1:(system.control_order + 1)
        MOI.add_constraints(
            optimizer,
            variables[n_wfn_states + i],
            MOI.EqualTo(0.0)
        )
    end

    # final a(t = T) constraints: ∫a, a, da = 0
    for i = 1:(system.control_order + 1)
        MOI.add_constraints(
            optimizer,
            variables[index(T, n_wfn_states + i, vardim)],
            MOI.EqualTo(0.0)
        )
    end

    # bound |a(t)| < a_bound
    if bound_a
        for t = 2:T-1
            idx = index(t, n_wfn_states + 2, vardim)

            MOI.add_constraints(
                optimizer,
                variables[idx],
                MOI.LessThan(a_bound)
            )

            MOI.add_constraints(
                optimizer,
                variables[idx],
                MOI.GreaterThan(-a_bound)
            )
        end
    end

    traj = TrajectoryData(system, T, Δt; σ=σ)

    dynamics_constraints = [
        MOI.NLPBoundsPair(0.0, 0.0) for _ = 1:total_dynamics
    ]

    block_data = MOI.NLPBlockData(dynamics_constraints, evaluator, true)

    MOI.set(optimizer, MOI.NLPBlock(), block_data)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    return QubitProblem(
        system,
        evaluator,
        variables,
        optimizer,
        traj,
        T,
        Δt,
        vardim,
        total_vars,
        total_states,
        total_dynamics,
    )
end

function initialize_trajectory!(prob::QubitProblem, traj::TrajectoryData)
    vardim = prob.system.nstates + 1
    for (t, x, u) in zip(1:prob.T, traj.states, traj.controls)
        MOI.set(
            prob.optimizer,
            MOI.VariablePrimalStart(),
            prob.variables[slice(t, vardim)],
            [x; u]
        )
    end
end

initialize_trajectory!(prob::QubitProblem) = initialize_trajectory!(prob, prob.trajectory)

@views function update_traj_data!(prob::QubitProblem)
    vardim = prob.system.nstates + 1
    z = MOI.get(prob.optimizer, MOI.VariablePrimal(), prob.variables)
    xs = [z[slice(t, vardim; stretch=-1)] for t = 1:prob.T]
    us = [[z[index(t, vardim)]] for t = 1:prob.T]
    prob.trajectory.states .= xs
    prob.trajectory.controls .= us
end

function get_traj_data(prob::QubitProblem)
    update_traj_data!(prob)
    return prob.trajectory
end

function solve!(prob::QubitProblem)
    initialize_trajectory!(prob)
    MOI.optimize!(prob.optimizer)
    update_traj_data!(prob)
end


#
# min time problem
#

struct MinTimeProblem
    subprob::QubitProblem
    evaluator::MinTimeEvaluator
    variables::Vector{MOI.VariableIndex}
    optimizer::Ipopt.Optimizer
    T::Int
end

function MinTimeProblem(
    system::AbstractQubitSystem{N},
    T::Int;
    Rᵤ=0.001,
    Rₛ=0.001,
    integrator=pade_schroedinger,
    eval_hessian=false,
    bound_a=true,
    a_bound=1.0,
    min_time_options=Options(),
    kwargs...
) where N

    optimizer = Ipopt.Optimizer()

    # set Ipopt optimizer options
    for name in fieldnames(typeof(min_time_options))
        optimizer.options[String(name)] = getfield(min_time_options, name)
    end

    vardim = system.nstates + 1
    total_vars = vardim * T
    total_dynamics = system.nstates * (T - 1)

    # set up sub problem
    subprob = QubitProblem(
        system,
        T;
        integrator=integrator,
        eval_hessian=eval_hessian,
        bound_a=bound_a,
        a_bound=a_bound,
        kwargs...
    )

    # build min time evaluator
    evaluator = MinTimeEvaluator(
        subprob.system,
        integrator,
        T,
        Rᵤ,
        Rₛ,
        eval_hessian
    )

    variables = MOI.add_variables(optimizer, total_vars + T - 1)

    n_wfn_states = system.nqstates * system.isodim

    # initial quantum state constraints: ψ̃(t=1) = ψ̃1
    for i = 1:n_wfn_states
        MOI.add_constraints(
            optimizer,
            variables[i],
            MOI.EqualTo(system.ψ̃1[i])
        )
    end

    # initial a(t = 1 * Δt) constraints: ∫a, a, da = 0
    for i = 1:(system.control_order + 1)
        MOI.add_constraints(
            optimizer,
            variables[n_wfn_states + i],
            MOI.EqualTo(0.0)
        )
    end

    # final a(t = T * Δt) constraints: ∫a, a, da = 0
    for i = 1:(system.control_order + 1)
        MOI.add_constraints(
            optimizer,
            variables[index(T, n_wfn_states + i, vardim + 1)],
            MOI.EqualTo(0.0)
        )
    end

    # constraints on Δtₜs
    for t = 1:T-1
        idx = index(t, vardim + 1)

        MOI.add_constraints(
            optimizer,
            variables[idx],
            MOI.GreaterThan(0.1 * subprob.Δt)
        )

        MOI.add_constraints(
            optimizer,
            variables[idx],
            MOI.LessThan(2.0 * subprob.Δt)
        )
    end

    # bound |a(t)| < a_bound
    if bound_a
        for t = 2:T-1
            idx = index(t, n_wfn_states + 2, vardim + 1)

            MOI.add_constraints(
                optimizer,
                variables[idx],
                MOI.LessThan(a_bound)
            )

            MOI.add_constraints(
                optimizer,
                variables[idx],
                MOI.GreaterThan(-a_bound)
            )
        end
    end

    dynamics_constraints = [
        MOI.NLPBoundsPair(0.0, 0.0) for _ = 1:total_dynamics
    ]

    block_data = MOI.NLPBlockData(dynamics_constraints, evaluator, true)

    MOI.set(optimizer, MOI.NLPBlock(), block_data)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    return MinTimeProblem(
        subprob,
        evaluator,
        variables,
        optimizer,
        T,
    )
end

function initialize_trajectory!(prob::MinTimeProblem, traj::TrajectoryData)
    vardim = prob.subprob.system.nstates + 1
    Δt = prob.subprob.Δt
    for (t, x, u) in zip(1:prob.T, traj.states, traj.controls)
        MOI.set(
            prob.optimizer,
            MOI.VariablePrimalStart(),
            prob.variables[slice(t, t != prob.T ? vardim + 1 : vardim)],
            t != prob.T ? [x; u; Δt] : [x; u]
        )
    end
end

function initialize_trajectory!(prob::MinTimeProblem)
    initialize_trajectory!(prob, prob.subprob.trajectory)
end

@views function update_traj_data!(prob::MinTimeProblem)
    vardim = prob.subprob.system.nstates + 1
    z = MOI.get(prob.optimizer, MOI.VariablePrimal(), prob.variables)
    xs = [z[slice(t, vardim + 1; stretch=-2)] for t = 1:prob.T]
    us = [[z[index(t, vardim, vardim + 1)]] for t = 1:prob.T]
    Δts = [0.0; [z[index(t, vardim + 1)] for t = 1:prob.T-1]]
    prob.subprob.trajectory.states .= xs
    prob.subprob.trajectory.controls .= us
    prob.subprob.trajectory.times .= cumsum(Δts)
end


function solve!(prob::MinTimeProblem)
    solve!(prob.subprob)

    init_traj = get_traj_data(prob.subprob)

    initialize_trajectory!(prob, init_traj)

    # constrain end points to match subprob solution
    for j = 1:prob.subprob.system.nqstates*prob.subprob.system.isodim
        MOI.add_constraint(
            prob.optimizer,
            prob.variables[index(prob.T, j, prob.subprob.vardim + 1)],
            # prob.variables[end - prob.subprob.system.nstates - 1 + j],
            MOI.EqualTo(init_traj.states[end][j])
        )
    end

    MOI.optimize!(prob.optimizer)
    update_traj_data!(prob)
end



end
