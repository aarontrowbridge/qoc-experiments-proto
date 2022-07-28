module Trajectories

using ..Utils
using ..QubitSystems
using ..QuantumLogic

export Trajectory

export jth_order_controls
export jth_order_controls_matrix
export controls_matrix
export actions_matrix
export wfn_components
export wfn_components_matrix
export final_state
export final_state2


struct Trajectory
    states::Vector{Vector{Float64}}
    actions::Vector{Vector{Float64}}
    times::Vector{Float64}
    T::Int
end

# TODO: evaluate integrals of random control variable for augmented state

function Trajectory(
    system::AbstractQubitSystem,
    controls::Matrix,
    Δt::Real
)
    T = size(controls, 2)

    times = [Δt * t for t = 0:T-1]

    ∫controls = similar(controls)

    for k = 1:system.ncontrols
        ∫controls[k, :] = integral(controls[k, :], times)
    end

    augmented_controls = zeros(system.ncontrols * (system.augdim + 1), T)

    augmented_controls[slice(1, system.ncontrols), :] = ∫controls
    augmented_controls[slice(2, system.ncontrols), :] = controls

    for j = 1:system.control_order
        for k = 1:system.ncontrols

            aug_j_idx = index(
                2 + j,
                k,
                system.ncontrols
            )

            aug_j_minus_1_idx = index(
                2 + j - 1,
                k,
                system.ncontrols
            )

            augmented_controls[aug_j_idx, :] = derivative(
                augmented_controls[aug_j_minus_1_idx, :],
                times
            )
        end
    end

    augs_matrix =
        augmented_controls[1:system.ncontrols*system.augdim, :]

    actions_matrix =
        augmented_controls[end-system.ncontrols+1:end, :]

    states = []

    push!(states, [system.ψ̃1; zeros(system.n_aug_states)])

    for t = 2:T-1
        augs = augs_matrix[:, t]
        wfns = zeros(system.n_wfn_states)
        state = [wfns; augs]
        push!(states, state)
    end

    push!(states, [system.ψ̃goal; zeros(system.n_aug_states)])

    actions = [actions_matrix[:, t] for t = 1:T]

    return Trajectory(states, actions, times, T)
end

function Trajectory(system::AbstractQubitSystem, Δt::Float64, T::Int)
    states = []
    push!(states, [system.ψ̃1; zeros(system.n_aug_states)])
    for t = 2:T-1
        state1 = normalize([1. - t/T, 0, 0, 0, -t/T, 0])
        print(state1)
        state2 = normalize([0, 1 - t/T, 0, -t/T, 0, 0])
        wfns = vcat(state1, state2)
        #wfns = zeros(system.n_wfn_states)
        augs = randn(system.n_aug_states)
        state = [wfns; augs]
        push!(states, state)
    end
    push!(states, [system.ψ̃goal; zeros(system.n_aug_states)])

    actions = [[0.1*randn(system.ncontrols) for t = 1:T-1]..., zeros(system.ncontrols)]

    times = [Δt * t for t = 1:T]

    return Trajectory(states, actions, times, T)
end

function derivative(xs::Vector, ts::Vector)
    dxs = similar(xs)
    dxs[1] = 0.0
    for t = 2:length(ts)
        Δt = ts[t] - ts[t - 1]
        Δx = xs[t] - xs[t - 1]
        dxs[t] = Δx / Δt
    end
    return dxs
end

function integral(xs::Vector, ts::Vector)
    ∫xs = similar(xs)
    ∫xs[1] = xs[1] * (ts[2] - ts[1])
    for t = 2:length(ts)
        Δt = ts[t] - ts[t - 1]
        ∫xs[t] = ∫xs[t - 1] + xs[t] * Δt
    end
    return ∫xs
end

function jth_order_controls(traj::Trajectory, sys::AbstractQubitSystem, j::Int)
    @assert j ∈ -1:sys.control_order
    if j != sys.control_order
        jth_order_slice = slice(2 + j, sys.ncontrols)
        return [traj.states[t][sys.n_wfn_states .+ (jth_order_slice)] for t = 1:traj.T]
    else
        # returns same value for T-1 and T to make plots cleaner
        return [traj.actions[1:end-1]..., traj.actions[end-1]]
    end
end

#get rid of the 2 because no ∫a
function jth_order_controls(traj::Trajectory, sys::TransmonSystem, j::Int)
    @assert j ∈ 0:sys.control_order
    if j != sys.control_order
        jth_order_slice = slice(1 + j, sys.ncontrols)
        return [traj.states[t][sys.n_wfn_states .+ (jth_order_slice)]/(2π) for t = 1:traj.T]
    else
        return traj.actions
    end
end

# function jth_order_controls(traj::Trajectory, sys::MultiModeSystem, j::Int)
#     @assert j ∈ 0:sys.control_order
#     if j != sys.control_order
#         jth_order_slice = slice(j + 1, sys.ncontrols)
#         return [traj.states[t][sys.n_wfn_states .+ jth_order_slice]/(2π) for t = 1:traj.T]
#     else
#         return traj.actions
#     end
# end

function jth_order_controls_matrix(traj, sys, j)
    return hcat(jth_order_controls(traj, sys, j)...)
end

controls_matrix(traj, sys) = jth_order_controls_matrix(traj, sys, 0)

actions_matrix(traj::Trajectory) = hcat(traj.actions...)

function wfn_components(
    traj::Trajectory,
    sys::AbstractQubitSystem;
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

function final_state2(
    traj::Trajectory,
    sys::AbstractQubitSystem;
)
    return traj.states[traj.T][slice(1, sys.isodim + 1, 2*sys.isodim, sys.isodim)]
end

function final_state(
    traj::Trajectory,
    sys::AbstractQubitSystem;
)
    return traj.states[traj.T][slice(1, sys.isodim)]
end

# function populations(
#     traj::Trajectory,
#     sys::AbstractQubitSystem,
#     i = 1,
#     components=nothing
# )

wfn_components_matrix(args...; kwargs...) = hcat(wfn_components(args...; kwargs...)...)

end
