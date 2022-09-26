module Trajectories

using ..Utils
using ..QuantumSystems
using ..QuantumLogic
using ..Integrators

using HDF5

export Trajectory

export jth_order_controls
export jth_order_controls_matrix
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

#
# helper functions
#

function linear_interpolation(x1, xT, T::Int)
    X = Vector{typeof(x1)}(undef, T)
    Δx = xT - x1
    for t = 1:T
        X[t] = x1 + Δx * (t - 1) / (T - 1)
    end
    return X
end

function rollout(
    sys::AbstractSystem,
    A::Vector{<:AbstractVector},
    Δt::Real
)
    T = length(A) + 1
    Ψ̃ = Vector{typeof(sys.ψ̃goal)}(undef, T)
    Ψ̃[1] = sys.ψ̃goal
    for t = 2:T
        Gₜ = G(A[t - 1], sys.G_drift, sys.G_drives)
        ψ̃ⁱₜ₋₁s = @views [
            Ψ̃[t - 1][slice(i, sys.isodim)]
                for i = 1:sys.nqstates
        ]
        Uₜ = exp(Gₜ * Δt)
        Ψ̃[t] = vcat([Uₜ * ψ̃ⁱₜ₋₁ for ψ̃ⁱₜ₋₁ in ψ̃ⁱₜ₋₁s]...)
    end
    return Ψ̃
end

function derivative(xs::Vector, ts::Vector)
    dxs = similar(xs)
    dxs[1] = 0.0
    for t = 2:length(ts)
        Δx = xs[t] - xs[t - 1]
        Δt = ts[t] - ts[t - 1]
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





struct Trajectory
    states::Vector{Vector{Float64}}
    actions::Vector{Vector{Float64}}
    times::Vector{Float64}
    T::Int
    Δt::Float64
end

function Base.:+(traj1::Trajectory, traj2::Trajectory)
    states = traj1.states .+ traj2.states
    actions = traj1.actions .+ traj2.actions
    return Trajectory(
        states,
        actions,
        traj1.times,
        traj1.T,
        traj1.Δt
    )
end

# constructor for case of given controls

function Trajectory(
    sys::AbstractSystem,
    controls::Matrix,
    Δt::Real
)
    T = size(controls, 2) + 1

    times = [Δt * t for t = 1:T]

    controls = hcat(controls, controls[:, end])

    if sys.∫a
        ∫controls = similar(controls)

        for k = 1:sys.ncontrols
            ∫controls[k, :] = integral(controls[k, :], times)
        end
    end

    augss = zeros(sys.ncontrols * (sys.augdim + 1), T)

    if sys.∫a
        augss[slice(1, sys.ncontrols), :] = ∫controls
    end

    augss[slice(sys.∫a + 1, sys.ncontrols), :] = controls

    for j = 1:sys.control_order
        for k = 1:sys.ncontrols

            aug_j_idx = index(
                sys.∫a + 1 + j,
                k,
                sys.ncontrols
            )

            aug_j_minus_1_idx = index(
                sys.∫a + 1 + j - 1,
                k,
                sys.ncontrols
            )

            augss[aug_j_idx, :] = derivative(
                augss[aug_j_minus_1_idx, :],
                times
            )
        end
    end

    augs_matrix = augss[1:sys.n_aug_states, :]

    A = @views [vec(controls[:, t]) for t = 1:T-1]

    Ψ̃ = rollout(sys, A, Δt)


    states = []

    for t = 1:T
        wfns = Ψ̃[t]
        augs = augs_matrix[:, t]
        x = [wfns; augs]
        push!(states, x)
    end

    actions_matrix =
        augss[slice(sys.augdim + 1, sys.ncontrols), :]

    actions = [actions_matrix[:, t] for t = 1:T]

    return Trajectory(states, actions, times, T, Δt)
end

function Trajectory(
    sys::AbstractSystem,
    T::Int,
    Δt::Float64;
    linearly_interpolate = true,
    σ = 0.1
)

    if linearly_interpolate
        Ψ̃ = linear_interpolation(sys.ψ̃init, sys.ψ̃goal, T)
    else
        Ψ̃ = fill(2*rand(sys.n_wfn_states) - 1, T)
    end

    states = []

    push!(states, [sys.ψ̃init; zeros(sys.n_aug_states)])

    for t = 2:T-1
        wfns = Ψ̃[t]
        augs = σ * randn(sys.n_aug_states)
        x = [wfns; augs]
        push!(states, x)
    end

    push!(states, [sys.ψ̃goal; zeros(sys.n_aug_states)])

    actions = [
        [σ * randn(sys.ncontrols) for t = 1:T-1]...,
        zeros(sys.ncontrols)
    ]

    times = [Δt * t for t = 1:T]

    return Trajectory(states, actions, times, T, Δt)
end

function jth_order_controls(
    traj::Trajectory,
    sys::AbstractSystem,
    j::Int;
    d2pi=true
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

controls_matrix(traj, sys) =
    jth_order_controls_matrix(traj, sys, 0)

actions_matrix(traj::Trajectory) = hcat(traj.actions...)

function wfn_components(
    traj::Trajectory,
    sys::AbstractSystem;
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
    sys::AbstractSystem;
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
    sys::AbstractSystem,
    i::Int
)
    return traj.states[traj.T][slice(i, sys.isodim)]
end

# function populations(
#     traj::Trajectory,
#     sys::AbstractSystem,
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
    sys::AbstractSystem
)
    traj = load_trajectory(path)
    controls = controls_matrix(traj, sys)
    return controls, traj.times
end

end
