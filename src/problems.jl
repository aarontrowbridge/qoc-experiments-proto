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

function QubitProblem(
    system::AbstractQubitSystem,
    init_traj::Trajectory;
    kwargs...
)
    return QubitProblem(
        system,
        init_traj.T;
        init_traj=init_traj,
        kwargs...
    )
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
    a_bound = 2π * 19e-3,
    init_traj=Trajectory(system, Δt, T),
    options=Options(),
    return_constraints = false,
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
        system.n_wfn_states + system.∫a*system.ncontrols .+ (1:system.ncontrols),
        a_bound,
        system.vardim
    )
    push!(cons, a_bound_con)

    # TODO: fix constraints: a_bounds not working

    constrain!(optimizer, variables, cons)

    dynamics_constraints =
        fill(MOI.NLPBoundsPair(0.0, 0.0), total_dynamics)

    block_data =
        MOI.NLPBlockData(dynamics_constraints, evaluator, true)

    MOI.set(optimizer, MOI.NLPBlock(), block_data)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    prob = QubitProblem(
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

    if return_constraints
        return prob, cons
    else
        return prob
    end
end


# function QubitProblem(
#     system::MultiModeQubitSystem,
#     T::Int;
#     integrator=pade_schroedinger,
#     loss=amplitude_loss,
#     Δt=0.01,
#     Q=0.0,
#     Qf=200.0,
#     R=0.1,
#     eval_hessian=false,
#     pin_first_qstate=true,
#     a_bound=1.0,
#     init_traj=Trajectory(system, Δt, T),
#     options=Options(),
# )

#     if getfield(options, :linear_solver) == "pardiso"
#         Libdl.dlopen("/usr/lib/liblapack.so.3", RTLD_GLOBAL)
#         Libdl.dlopen("/usr/lib/libomp.so", RTLD_GLOBAL)
#     end

#     optimizer = Ipopt.Optimizer()

#     # set Ipopt optimizer options
#     for name in fieldnames(typeof(options))
#         optimizer.options[String(name)] = getfield(options, name)
#     end

#     total_vars = system.vardim * T
#     total_dynamics = system.nstates * (T - 1)
#     total_states = system.nstates * T

#     variables = MOI.add_variables(optimizer, total_vars)

#     evaluator = QubitEvaluator(
#         system,
#         integrator,
#         loss,
#         eval_hessian,
#         T, Δt,
#         Q, R
#     )

#     cons = AbstractConstraint[]

#     # initial quantum state constraints: ψ̃(t=1) = ψ̃1
#     ψ1_con = EqualityConstraint(
#         1,
#         1:system.n_wfn_states,
#         system.ψ̃1,
#         system.vardim
#     )
#     push!(cons, ψ1_con)

#     # pin first qstate to be equal to analytic solution
#     if pin_first_qstate
#         pin_con = EqualityConstraint(
#             T,
#             1:system.isodim,
#             system.ψ̃goal[1:system.isodim],
#             system.vardim
#         )
#         push!(cons, pin_con)
#     end

#     # initial a(t = 1) constraints: ∫a, a, da = 0
#     aug1_con = EqualityConstraint(
#         1,
#         system.n_wfn_states .+ (1:system.n_aug_states),
#         0.0,
#         system.vardim
#     )
#     push!(cons, aug1_con)

#     # final a(t = T) constraints: ∫a, a, da = 0
#     augT_con = EqualityConstraint(
#         T,
#         system.n_wfn_states .+ (1:system.n_aug_states),
#         0.0,
#         system.vardim
#     )
#     push!(cons, augT_con)


#     # bound |a(t)| < a_bound
#     a_bound_con = BoundsConstraint(
#         2:T-1,
#         system.n_wfn_states .+ (1:system.ncontrols),
#         a_bound,
#         system.vardim
#     )
#     push!(cons, a_bound_con)

#     constrain!(optimizer, variables, cons)

#     dynamics_constraints = fill(MOI.NLPBoundsPair(0.0, 0.0), total_dynamics)

#     block_data = MOI.NLPBlockData(dynamics_constraints, evaluator, true)

#     MOI.set(optimizer, MOI.NLPBlock(), block_data)
#     MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

#     return QubitProblem(
#         system,
#         evaluator,
#         variables,
#         optimizer,
#         init_traj,
#         T,
#         Δt,
#         total_vars,
#         total_states,
#         total_dynamics,
#     )
# end


function initialize_trajectory!(
    prob::QubitProblem,
    traj::Trajectory
)
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
    Z = MOI.get(
        prob.optimizer,
        MOI.VariablePrimal(),
        prob.variables
    )

    xs = []

    for t = 1:prob.T
        xₜ = Z[slice(t, prob.system.nstates, prob.system.vardim)]
        push!(xs, xₜ)
    end

    us = []

    for t = 1:prob.T
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
    Δt=0.01,
    Δt_lbound=0.1 * Δt,
    Δt_ubound=2.0 * Δt,
    integrator=:FourthOrderPade,
    eval_hessian=true,
    a_bound=1.0,
    min_time_options=Options(),
    kwargs...
)

    optimizer = Ipopt.Optimizer()

    # set Ipopt optimizer options
    for name in fieldnames(typeof(min_time_options))
        optimizer.options[String(name)] =
            getfield(min_time_options, name)
    end

    total_vars = system.vardim * T
    total_dynamics = system.nstates * (T - 1)

    # defining Z = [Z; Δts]
    variables = MOI.add_variables(optimizer, total_vars + T - 1)

    # set up sub problem
    subprob, cons = QubitProblem(
        system,
        T;
        return_constraints=true,
        integrator=integrator,
        eval_hessian=eval_hessian,
        a_bound=a_bound,
        kwargs...
    )

    # constraints on Δtₜs
    Δt_bounds_con =
        TimeStepBoundsConstraint((Δt_lbound, Δt_ubound), T)
    push!(cons, Δt_bounds_con)

    constrain!(optimizer, variables, cons)

    # build min time evaluator
    evaluator = MinTimeEvaluator(
        subprob.system,
        integrator,
        T,
        Rᵤ,
        Rₛ,
        eval_hessian
    )

    dynamics_constraints =
        [MOI.NLPBoundsPair(0.0, 0.0) for _ = 1:total_dynamics]

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

function initialize_trajectory!(
    prob::MinTimeProblem,
    traj::Trajectory
)
    for (t, x, u) in zip(1:prob.T, traj.states, traj.actions)
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
        prob.variables[(end - (prob.T - 1) + 1):end],
        fill(prob.subprob.Δt, prob.T - 1)
    )
end

function initialize_trajectory!(prob::MinTimeProblem)
    initialize_trajectory!(prob, prob.subprob.trajectory)
end

@views function update_traj_data!(prob::MinTimeProblem)

    vardim  = prob.subprob.system.vardim
    nstates = prob.subprob.system.nstates

    Z = MOI.get(
        prob.optimizer,
        MOI.VariablePrimal(),
        prob.variables
    )

    xs = [Z[slice(t, nstates, vardim)] for t = 1:prob.T]

    us = [
        Z[slice(t, nstates + 1, vardim, vardim)] for t = 1:prob.T
    ]

    Δts = [0.0; Z[(end - (prob.T - 1) + 1):end]]

    prob.subprob.trajectory.states .= xs
    prob.subprob.trajectory.actions .= us
    prob.subprob.trajectory.times .= cumsum(Δts)
end


function solve!(prob::MinTimeProblem)
    solve!(prob.subprob)

    init_traj = prob.subprob.trajectory

    initialize_trajectory!(prob, init_traj)


    # constrain end points to match subprob solution

    isodim       = prob.subprob.system.isodim
    n_wfn_states = prob.subprob.system.n_wfn_states

    ψ̃T_con! = EqualityConstraint(
        prob.T,
        isodim+1:n_wfn_states,
        init_traj.states[end][isodim+1:n_wfn_states],
        prob.subprob.system.vardim
    )

    ψ̃T_con!(prob.optimizer, prob.variables)

    MOI.optimize!(prob.optimizer)

    update_traj_data!(prob)
end



end
