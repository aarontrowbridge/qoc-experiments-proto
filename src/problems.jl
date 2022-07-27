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
using ..Constraints

using Ipopt
using Libdl
using MathOptInterface
const MOI = MathOptInterface


#
# qubit problem
#

struct QubitProblem
    system::AbstractQubitSystem
    evaluator::QubitEvaluator
    variables::Vector{MOI.VariableIndex}
    optimizer::Ipopt.Optimizer
    trajectory::Trajectory
    T::Int
    Δt::Float64
    total_vars::Int
    total_states::Int
    total_dynamics::Int
end

# function QubitProblem(mmsys::MultiModeQubitSystem; kwargs...)
#     T = length(mmsys.ts)
#     Δt = mmsys.ts[2] - mmsys.ts[1]
#     return QubitProblem(mmsys, T; Δt=Δt, kwargs...)
# end

function QubitProblem(system::AbstractQubitSystem, init_traj::Trajectory; kwargs...)
    return QubitProblem(system, init_traj.T; init_traj=init_traj, kwargs...)
end

function QubitProblem(
    system::AbstractQubitSystem,
    T::Int;
    integrator=:FourthOrderPade,
    loss=amplitude_loss,
    Δt=0.01,
    Q=200.0,
    R=0.1,
    eval_hessian=true,
    pin_first_qstate=true,
    a_bound=1.0,
    init_traj=Trajectory(system, Δt, T),
    options=Options(),
)

    if getfield(options, :linear_solver) == "pardiso" &&
        !Sys.isapple()

        Libdl.dlopen("/usr/lib/liblapack.so.3", RTLD_GLOBAL)
        Libdl.dlopen("/usr/lib/libomp.so", RTLD_GLOBAL)
    end

    optimizer = Ipopt.Optimizer()

    # set Ipopt optimizer options
    for name in fieldnames(typeof(options))
        optimizer.options[String(name)] = getfield(options, name)
    end

    total_vars = system.vardim * T
    total_dynamics = system.nstates * (T - 1)
    total_states = system.nstates * T

    variables = MOI.add_variables(optimizer, total_vars)

    # TODO: this is super hacky, I know; it should be fixed
    # subtype(::Type{Type{T}}) where T <: AbstractQuantumIntegrator = T
    # integrator = subtype(integrator)

    evaluator = QubitEvaluator(
        system,
        integrator,
        loss,
        eval_hessian,
        T, Δt,
        Q, R
    )

    cons = AbstractConstraint[]

    # initial quantum state constraints: ψ̃(t=1) = ψ̃1
    ψ1_con = EqualityConstraint(
        1,
        1:system.n_wfn_states,
        system.ψ̃1,
        system.vardim
    )
    push!(cons, ψ1_con)

    # pin first qstate to be equal to analytic solution
    if pin_first_qstate
        pin_con = EqualityConstraint(
            T,
            1:system.isodim,
            system.ψ̃goal[1:system.isodim],
            system.vardim
        )
        push!(cons, pin_con)
    end

    # initial a(t = 1) constraints: ∫a, a, da = 0
    aug1_con = EqualityConstraint(
        1,
        system.n_wfn_states .+ (1:system.n_aug_states),
        0.0,
        system.vardim
    )
    push!(cons, aug1_con)

    # final a(t = T) constraints: ∫a, a, da = 0
    augT_con = EqualityConstraint(
        T,
        system.n_wfn_states .+ (1:system.n_aug_states),
        0.0,
        system.vardim
    )
    push!(cons, augT_con)


    # bound |a(t)| < a_bound
    a_bound_con = BoundsConstraint(
        2:T-1,
        system.n_wfn_states + system.ncontrols .+
            (1:system.ncontrols),
        a_bound,
        system.vardim
    )
    push!(cons, a_bound_con)

    # TODO: fix constraints: a_bounds not working

    optimizer = constrain(optimizer, variables, cons)

    dynamics_constraints = fill(MOI.NLPBoundsPair(0.0, 0.0), total_dynamics)

    block_data = MOI.NLPBlockData(dynamics_constraints, evaluator, true)

    MOI.set(optimizer, MOI.NLPBlock(), block_data)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    return QubitProblem(
        system,
        evaluator,
        variables,
        optimizer,
        init_traj,
        T,
        Δt,
        total_vars,
        total_states,
        total_dynamics,
    )
end


function QubitProblem(
    system::TransmonSystem,
    T::Int;
    integrator=pade_schroedinger,
    loss=amplitude_loss,
    Δt=0.01,
    Q=0.0,
    Qf=200.0,
    R=0.1,
    eval_hessian=false,
    pin_first_qstate=true,
    a_bound=1.0,
    init_traj=Trajectory(system, Δt, T),
    options=Options(),
)

    if getfield(options, :linear_solver) == "pardiso"
        Libdl.dlopen("/usr/lib/liblapack.so.3", RTLD_GLOBAL)
        Libdl.dlopen("/usr/lib/libomp.so", RTLD_GLOBAL)
    end

    optimizer = Ipopt.Optimizer()

    # set Ipopt optimizer options
    for name in fieldnames(typeof(options))
        optimizer.options[String(name)] = getfield(options, name)
    end

    total_vars = system.vardim * T
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

    cons = AbstractConstraint[]

    # initial quantum state constraints: ψ̃(t=1) = ψ̃1
    ψ1_con = EqualityConstraint(
        1,
        1:system.n_wfn_states,
        system.ψ̃1,
        system.vardim
    )
    push!(cons, ψ1_con)

    # pin first qstate to be equal to analytic solution
    if pin_first_qstate
        pin_con = EqualityConstraint(
            T,
            1:system.isodim,
            system.ψ̃f[1:system.isodim],
            system.vardim
        )
        push!(cons, pin_con)
    end

    # initial a(t = 1) constraints: ∫a, a, da = 0
    aug1_con = EqualityConstraint(
        1,
        system.n_wfn_states .+ (1:system.n_aug_states),
        0.0,
        system.vardim
    )
    push!(cons, aug1_con)

    # final a(t = T) constraints: ∫a, a, da = 0
    augT_con = EqualityConstraint(
        T,
        system.n_wfn_states .+ (1:system.n_aug_states),
        0.0,
        system.vardim
    )
    push!(cons, augT_con)


    # bound |a(t)| < a_bound
    a_bound_con = BoundsConstraint(
        2:T-1,
        system.n_wfn_states .+ (1:system.ncontrols),
        a_bound,
        system.vardim
    )
    push!(cons, a_bound_con)

    constrain!(optimizer, variables, cons)

    dynamics_constraints = fill(MOI.NLPBoundsPair(0.0, 0.0), total_dynamics)

    block_data = MOI.NLPBlockData(dynamics_constraints, evaluator, true)

    MOI.set(optimizer, MOI.NLPBlock(), block_data)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    return QubitProblem(
        system,
        evaluator,
        variables,
        optimizer,
        init_traj,
        T,
        Δt,
        total_vars,
        total_states,
        total_dynamics,
    )
end


function QubitProblem(
    system::MultiModeQubitSystem,
    T::Int;
    integrator=pade_schroedinger,
    loss=amplitude_loss,
    Δt=0.01,
    Q=0.0,
    Qf=200.0,
    R=0.1,
    eval_hessian=false,
    pin_first_qstate=true,
    a_bound=1.0,
    init_traj=Trajectory(system, Δt, T),
    options=Options(),
)

    if getfield(options, :linear_solver) == "pardiso"
        Libdl.dlopen("/usr/lib/liblapack.so.3", RTLD_GLOBAL)
        Libdl.dlopen("/usr/lib/libomp.so", RTLD_GLOBAL)
    end

    optimizer = Ipopt.Optimizer()

    # set Ipopt optimizer options
    for name in fieldnames(typeof(options))
        optimizer.options[String(name)] = getfield(options, name)
    end

    total_vars = system.vardim * T
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

    cons = AbstractConstraint[]

    # initial quantum state constraints: ψ̃(t=1) = ψ̃1
    ψ1_con = EqualityConstraint(
        1,
        1:system.n_wfn_states,
        system.ψ̃1,
        system.vardim
    )
    push!(cons, ψ1_con)

    # pin first qstate to be equal to analytic solution
    if pin_first_qstate
        pin_con = EqualityConstraint(
            T,
            1:system.isodim,
            system.ψ̃f[1:system.isodim],
            system.vardim
        )
        push!(cons, pin_con)
    end

    # initial a(t = 1) constraints: ∫a, a, da = 0
    aug1_con = EqualityConstraint(
        1,
        system.n_wfn_states .+ (1:system.n_aug_states),
        0.0,
        system.vardim
    )
    push!(cons, aug1_con)

    # final a(t = T) constraints: ∫a, a, da = 0
    augT_con = EqualityConstraint(
        T,
        system.n_wfn_states .+ (1:system.n_aug_states),
        0.0,
        system.vardim
    )
    push!(cons, augT_con)


    # bound |a(t)| < a_bound
    a_bound_con = BoundsConstraint(
        2:T-1,
        system.n_wfn_states .+ (1:system.ncontrols),
        a_bound,
        system.vardim
    )
    push!(cons, a_bound_con)

    constrain!(optimizer, variables, cons)

    dynamics_constraints = fill(MOI.NLPBoundsPair(0.0, 0.0), total_dynamics)

    block_data = MOI.NLPBlockData(dynamics_constraints, evaluator, true)

    MOI.set(optimizer, MOI.NLPBlock(), block_data)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    return QubitProblem(
        system,
        evaluator,
        variables,
        optimizer,
        init_traj,
        T,
        Δt,
        total_vars,
        total_states,
        total_dynamics,
    )
end


function initialize_trajectory!(prob::QubitProblem, traj::Trajectory)
    for (t, x, u) in zip(1:prob.T, traj.states, traj.actions)
        MOI.set(
            prob.optimizer,
            MOI.VariablePrimalStart(),
            prob.variables[slice(t, prob.system.vardim)],
            [x; u]
        )
    end
end

initialize_trajectory!(prob) = initialize_trajectory!(prob, prob.trajectory)

@views function update_traj_data!(prob::QubitProblem)
    z = MOI.get(
        prob.optimizer,
        MOI.VariablePrimal(),
        prob.variables
    )

    xs = [
        z[slice(t, 1, prob.system.nstates, prob.system.vardim)]
            for t = 1:prob.T
    ]
    us = [
        z[
            slice(
                t,
                prob.system.nstates + 1,
                prob.system.vardim,
                prob.system.vardim
            )
        ] for t = 1:prob.T
    ]

    prob.trajectory.states .= xs
    prob.trajectory.actions .= us
end

function solve!(prob::QubitProblem; init_traj=prob.trajectory)
    initialize_trajectory!(prob, init_traj)
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
    system::AbstractQubitSystem,
    T::Int;
    Rᵤ=0.001,
    Rₛ=0.001,
    integrator=FourthOrderPade,
    eval_hessian=false,
    bound_a=true,
    a_bound=1.0,
    min_time_options=Options(),
    kwargs...
)

    optimizer = Ipopt.Optimizer()

    # set Ipopt optimizer options
    for name in fieldnames(typeof(min_time_options))
        optimizer.options[String(name)] = getfield(min_time_options, name)
    end

    total_vars = system.vardim * T
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

    # initial quantum state constraints: ψ̃(t=1) = ψ̃1
    for i = 1:system.n_wfn_states
        MOI.add_constraints(
            optimizer,
            variables[i],
            MOI.EqualTo(system.ψ̃1[i])
        )
    end

    # initial a(t = 1 * Δt) constraints: ∫a, a, da = 0
    for i = 1:system.n_aug_states
        MOI.add_constraints(
            optimizer,
            variables[system.n_wfn_states + i],
            MOI.EqualTo(0.0)
        )
    end

    # final a(t = T * Δt) constraints: ∫a, a, da = 0
    for i = 1:system.n_aug_states
        MOI.add_constraints(
            optimizer,
            variables[index(T, system.n_wfn_states + i, system.vardim + 1)],
            MOI.EqualTo(0.0)
        )
    end

    # constraints on Δtₜs
    for t = 1:T-1
        idx = index(t, system.vardim + 1)

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
            for k = 1:system.ncontrols

                idx = index(
                    t,
                    system.n_wfn_states + system.ncontrols + k,
                    system.vardim + 1
                )

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

function initialize_trajectory!(prob::MinTimeProblem, traj::Trajectory)
    vardim = prob.subprob.system.vardim
    Δt = prob.subprob.Δt
    for (t, x, u) in zip(1:prob.T, traj.states, traj.actions)
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
    vardim = prob.subprob.system.vardim
    nstates = prob.subprob.system.nstates
    z = MOI.get(prob.optimizer, MOI.VariablePrimal(), prob.variables)
    xs = [z[slice(t, 1, nstates, vardim + 1)] for t = 1:prob.T]
    us = [z[slice(t, nstates + 1, vardim, vardim + 1)] for t = 1:prob.T]
    Δts = [0.0; [z[index(t, vardim + 1)] for t = 1:prob.T-1]]
    prob.subprob.trajectory.states .= xs
    prob.subprob.trajectory.actions .= us
    prob.subprob.trajectory.times .= cumsum(Δts)
end


function solve!(prob::MinTimeProblem)
    solve!(prob.subprob)

    init_traj = prob.subprob.trajectory

    initialize_trajectory!(prob, init_traj)

    # constrain end points to match subprob solution
    for j = 1:prob.subprob.system.n_wfn_states
        MOI.add_constraint(
            prob.optimizer,
            prob.variables[index(prob.T, j, prob.subprob.system.vardim + 1)],
            MOI.EqualTo(init_traj.states[end][j])
        )
    end

    MOI.optimize!(prob.optimizer)
    update_traj_data!(prob)
end



end
