module Trajectories

using ..Utils
using ..QuantumSystems
using ..QuantumLogic
using ..Integrators

export dumb_downsample

using HDF5

export Trajectory

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
    Δt::Real;
    integrator=fourth_order_pade
)
    T = length(A) + 1
    Ψ̃ = Vector{typeof(sys.ψ̃goal)}(undef, T)
    Ψ̃[1] = sys.ψ̃goal
    for t = 2:T
        Gₜ = Integrators.G(A[t - 1], sys.G_drift, sys.G_drives)
        ψ̃ⁱₜ₋₁s = @views [
            Ψ̃[t - 1][slice(i, sys.isodim)]
                for i = 1:sys.nqstates
        ]
        Uₜ = integrator(Gₜ, Δt)
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

function dumb_downsample(traj::Trajectory, take_every_n::Int)
    @assert traj.T % take_every_n == 0
    Δt_new = traj.Δt * take_every_n
    states_new = traj.states[1:take_every_n:end]
    actions_new = traj.actions[1:take_every_n:end]
    times_new = traj.times[1:take_every_n:end]
    return Trajectory(
        states_new,
        actions_new,
        times_new,
        traj.T ÷ take_every_n,
        Δt_new
    )
end

end
