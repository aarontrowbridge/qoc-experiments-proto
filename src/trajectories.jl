module Trajectories

using ..QubitSystems

export TrajectoryData

struct TrajectoryData
    states::Vector{Vector{Float64}}
    actions::Vector{Vector{Float64}}
    times::Vector{Float64}
end

# TODO: evaluate integrals of random control variable for augmented state

function TrajectoryData(
    system::AbstractQubitSystem,
    T::Int,
    Δt::Float64;
    σ=1.0
)

    states = [
        [system.ψ̃1; zeros(system.n_aug_states)],
        [σ * randn(system.nstates) for _ in 2:T-1]...,
        [system.ψ̃f; zeros(system.n_aug_states)]
    ]

    actions = [
        zeros(system.ncontrols),
        [σ * randn(system.ncontrols) for _ in 2:T-1]...,
        zeros(system.ncontrols)
    ]

    times = [Δt * t for t = 0:T-1]

    return TrajectoryData(states, actions, times)
end

function TrajectoryData(
    system::AbstractQubitSystem,
    Δt::Real,
    controls::Matrix;
    σ=1.0
)
    T = size(controls, 2)

    states = [
        [system.ψ̃1; zeros(system.n_aug_states)],
        [[σ * randn(system.n_wfn_states);
          vcat([[0.0, controls[k, t], zeros(system.control_order - 1)...]
          for k = 1:system.ncontrols]...)
        ] for t in 2:T-1]...,
        [system.ψ̃f; zeros(system.n_aug_states)]
    ]

    actions = [
        zeros(system.ncontrols),
        [σ * randn(system.ncontrols) for _ in 2:T-1]...,
        zeros(system.ncontrols)
    ]

    times = [Δt * t for t = 0:T-1]

    return TrajectoryData(states, actions, times)
end

function augs_and_actions(traj::TrajectoryData, sys::AbstractQubitSystem)
    # [[[∫a₁, a₁(t), da₁(t), dda₁(t)], [∫a₂, a₂(t), da₂(t), dda₂(t)], ...] for t = 1:T]
    data = [
        [
            vcat(
                traj.states[t][
                    sys.n_wfn_states .+ slice(k, sys.augdim)
                ],
                traj.actions[t][k]
            )
            for k = 1:sys.ncontrols
        ] for t = 1:length(traj.times)
    ]
end

function controls(traj::TrajectoryData, sys::AbstractQubitSystem)
    return augs_and_actions(traj, sys)[2, :]
end




end
