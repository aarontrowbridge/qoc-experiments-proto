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
# using ..MinTimeProblems

using HDF5
using JLD2
using Ipopt

struct ProblemData
    system::AbstractSystem
    # objective_terms::Vector{Dict}
    # constraints::Vector{AbstractConstraint}
    trajectory::Trajectory
    params::Dict

    function ProblemData(prob::QuantumControlProblem)
        return new(
            prob.system,
            # prob.objective_terms,
            # prob.constraints,
            prob.trajectory,
            prob.params
        )
    end

    # function ProblemData(prob::MinTimeProblem)
    #     return new(
    #         :MinTime,
    #         prob.subprob.system,
    #         prob.objective_terms,
    #         prob.constraints,
    #         prob.subprob.trajectory,
    #         merge(prob.params, prob.subprob.params)
    #     )
    # end
end

function Problems.QuantumControlProblem(data::ProblemData)
    additional_objective_terms = pop!(data.params, :additional_objective_terms)
    if isempty(additional_objective_terms)
        additional_objective = nothing
    else
        additional_objective = Objective(additional_objective_terms)
    end
    return QuantumControlProblem(
        data.system;
        additional_objective=additional_objective,
        init_traj=data.trajectory,
        data.params...
    )
end

# function Problems.QuantumControlProblem(data::ProblemData)

#     objective = Objective(data.objective_terms)

#     dynamics = QuantumDynamics(
#         data.system,
#         data.params[:integrator],
#         data.params[:Z_indices],
#         data.params[:Δt_indices],
#         data.trajectory.T;
#         eval_hessian=data.params[:eval_hessian]
#     )

#     evaluator = PicoEvaluator(
#         objective,
#         dynamics,
#         data.params[:eval_hessian]
#     )

#     optimizer = Ipopt.Optimizer()

#     set!(optimizer, data.params[:options])

#     variables = Problems.initialize_optimizer!(
#         optimizer,
#         evaluator,
#         data.constraints,
#         data.params[:n_dynamics_constraints],
#         data.params[:n_variables]
#     )

#     return QuantumControlProblem(
#         data.system,
#         variables,
#         optimizer,
#         objective.terms,
#         data.constraints,
#         data.trajectory,
#         data.params
#     )
# end

function save_problem(prob::QuantumControlProblem, path::String)
    mkpath(dirname(path))
    data = ProblemData(prob)
    @save path data
end

function save_problem(data::ProblemData, path::String)
    mkpath(dirname(path))
    @save path data
end

function load_problem(path::String; kwargs...)

    @load path data

    for (key, value) in kwargs
        data.params[Symbol(key)] = value
    end

    # do this because Rᵤ and Rₛ depend on R in constructor
    if :R ∈ keys(kwargs)
        data.params[:Rᵤ] = data.params[:R]
        data.params[:Rₛ] = data.params[:R]
    end

    # do this to account for user defined objective
    additional_objective_terms = pop!(data.params, :additional_objective_terms)
    if isempty(additional_objective_terms)
        additional_objective = nothing
    else
        additional_objective = Objective(additional_objective_terms)
    end

    return QuantumControlProblem(
        data.system;
        additional_objective=additional_objective,
        init_traj=data.trajectory,
        data.params...
    )
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
