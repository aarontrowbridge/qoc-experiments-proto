module Problems

export QubitProblem
export solve!
export initialize_trajectory!
export update_traj_data!
export get_traj_data

using ..Integrators
using ..Losses
using ..QubitSystems
using ..Trajectories
using ..NLMOI
using ..IpoptOptions

using MathOptInterface
const MOI = MathOptInterface

import Ipopt


struct QubitProblem
    system::AbstractQubitSystem{N} where N
    evaluator::QubitEvaluator
    variables::Vector{MOI.VariableIndex}
    optimizer::Ipopt.Optimizer
    trajectory::TrajectoryData
    T::Int
    Δt::Float64
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
    options::Options=Options()
) where N

    optimizer = Ipopt.Optimizer()

    # set Ipopt optimizer options
    for name in fieldnames(typeof(options))
        optimizer.options[String(name)] = getfield(options, name)
    end

    total_vars = (system.nstates + N) * T
    total_dynamics = system.nstates * (T - 1)
    total_states = system.nstates * T

    variables = MOI.add_variables(optimizer, total_vars)

    dynamics_constraints = [MOI.NLPBoundsPair(-eps(), eps()) for _ = 1:total_dynamics]

    evaluator = QubitEvaluator(system, integrator, loss, eval_hessian, T, Δt, Q, Qf, R)

    n_wfn_states = system.nqstates * system.isodim

    # initial quantum state constraints: ψ̃(t=1) = ψ̃1
    for i = 1:n_wfn_states
        MOI.add_constraints(
            optimizer,
            variables[i],
            MOI.EqualTo(system.ψ̃1[i])
        )
    end

    # initial a(t = Tf) constraints: ∫a, a, da = 0
    for i = 1:(system.control_order + 1)
        MOI.add_constraints(
            optimizer,
            variables[n_wfn_states + i],
            MOI.EqualTo(0.0)
        )
    end

    # final a(t = 0) constraints: ∫a, a = 0
    for i = 1:system.control_order
        MOI.add_constraints(
            optimizer,
            variables[end - system.control_order - 2 + i],
            MOI.EqualTo(0.0)
        )
    end

    traj = TrajectoryData(system, T; σ=σ)

    constraints = dynamics_constraints

    blockdata = MOI.NLPBlockData(constraints, evaluator, true)

    MOI.set(optimizer, MOI.NLPBlock(), blockdata)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    return QubitProblem(
        system,
        evaluator,
        variables,
        optimizer,
        traj,
        T,
        Δt,
        total_vars,
        total_states,
        total_dynamics,
    )
end

function initialize_trajectory!(prob::QubitProblem, traj::TrajectoryData)
    vardim = prob.system.nstates + 1
    for (t, x, u) in zip(1:prob.T, traj.states, traj.actions)
        MOI.set(
            prob.optimizer,
            MOI.VariablePrimalStart(),
            prob.variables[(1+(t-1)*vardim):t*vardim],
            [x; u]
        )
    end
end

initialize_trajectory!(prob::QubitProblem) = initialize_trajectory!(prob, prob.trajectory)

@views function update_traj_data!(prob::QubitProblem)
    vardim = prob.system.nstates + 1
    y = MOI.get(prob.optimizer, MOI.VariablePrimal(), prob.variables)
    xus = [y[(1+(i-1)*vardim):(i*vardim)] for i in 1:prob.T]
    xs = [xu[1:end-1] for xu in xus]
    u = [xu[end:end] for xu in xus]
    prob.trajectory.states .= xs
    prob.trajectory.actions .= u
end

function get_traj_data(prob::QubitProblem)
    update_traj_data!(prob)
    return prob.trajectory
end

function solve!(prob::QubitProblem)
    initialize_trajectory!(prob)
    MOI.optimize!(prob.optimizer)
    update_traj_data!(prob)
    loss = MOI.get(prob.optimizer, MOI.ObjectiveValue())
    term_status = MOI.get(prob.optimizer, MOI.TerminationStatus())
    println()
    @info "solve completed" loss term_status
    println()
end


end
