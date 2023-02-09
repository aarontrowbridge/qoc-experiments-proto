module TrajectoryUtils

export jth_order_controls
export jth_order_controls_matrix
export controls
export controls_matrix
export actions_matrix
export wfn_components
export wfn_components_matrix
export final_state
export final_state_i
export final_state_2
export pop_components
export pop_matrix
export save_trajectory
export load_trajectory
export save_controls

using ..IndexingUtils
using ..QuantumUtils
using ..QuantumSystems
using ..Trajectories

using HDF5

function jth_order_controls(
    traj::Trajectory,
    sys::QuantumSystem,
    j::Int;
    d2pi=false
)
    if sys.∫a
        @assert j ∈ -1:sys.control_order
    else
        @assert j ∈ 0:sys.control_order
    end

    if j != sys.control_order
        jth_order_slice = slice(sys.∫a + 1 + j, sys.ncontrols)
        return [
            traj.states[t][
                sys.n_wfn_states .+ (jth_order_slice)
            ] for t = 1:traj.T
        ] / (1. + (2π - 1.) * d2pi)
    else
        # returns same value for actions at T-1 and T to make plots cleaner
        return [traj.actions[1:end-1]..., traj.actions[end-1]] /
            (1. + (2π - 1.) * d2pi)
    end
end



function jth_order_controls_matrix(traj, sys, j; kwargs...)
    return hcat(jth_order_controls(traj, sys, j; kwargs...)...)
end

controls(traj, sys; kwargs...) = jth_order_controls(traj, sys, 0; kwargs...)

controls_matrix(traj, sys; kwargs...) = hcat(controls(traj, sys; kwargs...)...)

actions_matrix(traj::Trajectory) = hcat(traj.actions...)

function wfn_components(
    traj::Trajectory,
    sys::QuantumSystem;
    i=1,
    components=nothing
)
    if isnothing(components)
        ψs = [traj.states[t][slice(i, sys.isodim)] for t = 1:traj.T]
    else
        ψs = [traj.states[t][index(i, 0, sys.isodim) .+ (components)] for t = 1:traj.T]
    end
    return ψs
end

function pop_components(
    traj::Trajectory,
    sys::QuantumSystem;
    i = 1,
    components = nothing
)
    if isnothing(components)
        pops = [abs2.(iso_to_ket(traj.states[t][slice(i, sys.isodim)])) for t = 1:traj.T]
    else
        pops = [abs2.(iso_to_ket(traj.states[t][slice(i, sys.isodim)])[components]) for t = 1:traj.T]
    end
    return pops
end



pop_matrix(args...; kwargs...) =
    hcat(pop_components(args...; kwargs...)...)

# get the second final state
final_state(traj, sys) = final_state_i(traj, sys, 1)
final_state_2(traj, sys) = final_state_i(traj, sys, 2)

function final_state_i(
    traj::Trajectory,
    sys::QuantumSystem,
    i::Int
)
    return traj.states[traj.T][slice(i, sys.isodim)]
end

# function populations(
#     traj::Trajectory,
#     sys::QuantumSystem,
#     i = 1,
#     components=nothing
# )

wfn_components_matrix(args...; kwargs...) =
    hcat(wfn_components(args...; kwargs...)...)


#
# saving and loading trajectories
#

function save_trajectory(traj::Trajectory, path::String)
    path_parts = split(path, "/")
    dir = joinpath(path_parts[1:end-1])
    if !isdir(dir)
        mkpath(dir)
    end
    h5open(path, "cw") do f
        f["states"] = hcat(traj.states...)
        f["actions"] = hcat(traj.actions...)
        f["times"] = traj.times
        f["T"] = traj.T
        f["dt"] = traj.Δt
    end
end

function load_trajectory(path::String)
    try
        h5open(path, "r") do f
            states_matrix = f["states"]
            actions_matrix = f["actions"]
            times = f["times"][]
            T = f["T"][]
            Δt = f["dt"][]
            states = [states_matrix[:, t] for t = 1:T]
            actions = [actions_matrix[:, t] for t = 1:T]
            return Trajectory(states, actions, times, T, Δt)
        end
    catch
        @warn "Could not load trajectory from file: " path
    end
end

function load_controls_matrix_and_times(
    path::String,
    sys::QuantumSystem
)
    traj = load_trajectory(path)
    controls = controls_matrix(traj, sys)
    return controls, traj.times
end

function save_controls(traj::Trajectory, sys::QuantumSystem, path::String)
    controls = controls_matrix(traj, sys)
    controls = transpose(controls) |> Array
    result = Dict(
        "total_time" => traj.times[end],
        "times" => traj.times,
        "T" => traj.T,
        "delta_t" => traj.Δt,
        "controls" => controls,
    )
    rm(path, force=true)
    mkpath(dirname(path))
    h5open(path, "cw") do f
        for (k, v) in result
            f[k] = v
        end
    end
end

# function load_trajectory_from_controls()


end
