module ProblemUtils

export ProblemData

export save_problem
export load_problem
export load_data
export get_and_save_controls
export generate_file_path

using ..IpoptOptions
using ..QuantumSystems
using ..Objectives
using ..Dynamics
using ..Evaluators
using ..Constraints
using ..Trajectories
using ..TrajectoryUtils
using ..Problems
using ..MinTimeProblems

using HDF5
using JLD2
using Ipopt

struct ProblemData <: AbstractProblem
    type::Symbol
    system::AbstractSystem
    objective_terms::Vector{Dict}
    constraints::Vector{AbstractConstraint}
    trajectory::Trajectory
    params::Dict

    function ProblemData(prob::FixedTimeProblem)
        return new(
            :FixedTime,
            prob.system,
            prob.objective_terms,
            prob.constraints,
            prob.trajectory,
            prob.params
        )
    end

    function ProblemData(prob::MinTimeProblem)
        return new(
            :MinTime,
            prob.subprob.system,
            prob.objective_terms,
            prob.constraints,
            prob.subprob.trajectory,
            merge(prob.params, prob.subprob.params)
        )
    end
end

function Problems.QuantumControlProblem(data::ProblemData)

    @assert data.type == :FixedTime "data is not from a FixedTimeProblem"

    objective = Objective(data.objective_terms)

    dynamics = QuantumDynamics(
        data.system,
        data.params[:integrator],
        data.trajectory.T,
        data.trajectory.Δt;
        eval_hessian=data.params[:eval_hessian]
    )

    evaluator = PicoEvaluator(
        objective,
        dynamics,
        data.params[:eval_hessian]
    )

    optimizer = Ipopt.Optimizer()

    set!(optimizer, data.params[:options])

    variables = Problems.initialize_optimizer!(
        optimizer,
        evaluator,
        data.constraints,
        data.params[:n_dynamics_constraints],
        data.params[:n_variables]
    )

    return QuantumControlProblem(
        data.system,
        variables,
        optimizer,
        objective.terms,
        data.constraints,
        data.trajectory,
        data.params
    )
end

function MinTimeProblems.MinTimeQuantumControlProblem(
    data::ProblemData
)

    @assert data.type == :MinTime "data is not from a MinTimeProblem"

    optimizer = Ipopt.Optimizer()

    set!(optimizer, data.params[:mintime_options])

    objective = Objective(data.objective_terms)

    dynamics = MinTimeQuantumDynamics(
        data.system,
        data.params[:mintime_integrator],
        data.params[:Z_indices],
        data.params[:Δt_indices],
        data.trajectory.T;
        eval_hessian=data.params[:mintime_eval_hessian]
    )

    evaluator = PicoEvaluator(
        objective,
        dynamics,
        data.params[:mintime_eval_hessian]
    )

    variables = Problems.initialize_optimizer!(
        optimizer,
        evaluator,
        data.constraints,
        data.params[:mintime_n_dynamics_constraints],
        data.params[:mintime_n_variables]
    )

    return MinTimeQuantumControlProblem(
        data,
        optimizer,
        variables,
        data.objective_terms,
        data.constraints,
        data.params
    )
end


function save_problem(prob::FixedTimeProblem, path::String)
    mkpath(dirname(path))
    data = ProblemData(prob)
    @save path data
end

function save_problem(prob::MinTimeProblem, path::String)
    mkpath(dirname(path))
    data = ProblemData(prob)
    merge!(data.params, prob.subprob.params)
    @save path data
end

function save_problem(data::ProblemData, path::String)
    mkpath(dirname(path))
    @save path data
end

function load_problem(path::String)
    @load path data
    if data.type == :FixedTime
        return QuantumControlProblem(data)
    elseif data.type == :MinTime
        return MinTimeQuantumControlProblem(data)
    else
        error("data is not from a FixedTimeProblem or MinTimeProblem")
    end
end

function load_data(path::String)::ProblemData
    @load path data
    return data
end

function get_and_save_controls(
    data_path::String,
    save_path::String
)
    data = load_data(data_path)
    save_controls(data.trajectory, data.system, save_path)
end


function MinTimeProblems.MinTimeQuantumControlProblem(;
    subprob_data=nothing,
    Rᵤ=0.001,
    Rₛ=0.001,
    Δt_lbound=0.2 * subprob_data.trajectory.Δt,
    Δt_ubound=1.1 * subprob_data.trajectory.Δt,
    mintime_eval_hessian=true,
    mintime_options=Options(),
    mintime_integrator=:FourthOrderPade,
    mintime_additional_objective=nothing,
    mintime_constraints=AbstractConstraint[]
)
    @assert !isnothing(subprob_data) "subproblem data must be provided"
    @assert subprob_data isa ProblemData "data must be of type ProblemData"
    @assert subprob_data.type == :FixedTime "data is not from a FixedTimeProblem"

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

    system = subprob_data.system
    T = subprob_data.trajectory.T

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

    Z_indices = 1:system.vardim * T
    Δt_indices = system.vardim * T .+ (1:T-1)

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
        subprob_data.constraints,
        mintime_constraints,
        Δt_con
    )

    mintime_n_dynamics_constraints =
        subprob_data.params[:n_dynamics_constraints]

    mintime_n_variables =
        subprob_data.params[:n_variables] + T - 1

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

    return MinTimeProblems.MinTimeQuantumControlProblem(
        subprob_data,
        optimizer,
        variables,
        objective.terms,
        cons,
        params
    )
end

function generate_file_path(extension, file_name, path)
    # Ensure the path exists.
    mkpath(path)

    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory.
    max_numeric_suffix = -1
    for (_, _, files) in walkdir(path)
        for file_name_ in files
            if occursin("$(file_name)", file_name_) && occursin(".$(extension)", file_name_)

                numeric_suffix = parse(
                    Int,
                    split(split(file_name_, "_")[end], ".")[1]
                )

                max_numeric_suffix = max(
                    numeric_suffix,
                    max_numeric_suffix
                )
            end
        end
    end

    file_path = joinpath(
        path,
        file_name *
        "_$(lpad(max_numeric_suffix + 1, 5, '0')).$(extension)"
    )

    return file_path
end




end
