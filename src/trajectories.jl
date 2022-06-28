module Trajectories

using ..QubitSystems

export TrajectoryData

struct TrajectoryData
    states::Vector{Vector{Float64}}
    actions::Vector{Vector{Float64}}
end

# TODO: do integrals of random control variable for augmented state

function TrajectoryData(system::AbstractQubitSystem{N}, T::Int; σ=1.0) where N
    states = [
        [system.ψ̃1; zeros(system.control_order + 1)],
        [σ * randn(system.nstates) for _ in 2:T-1]...,
        [system.ψ̃goal; zeros(system.control_order + 1)]
    ]
    actions = [
        zeros(N),
        [σ * randn(N) for _ in 2:T-1]...,
        zeros(N)
    ]
    return TrajectoryData(states, actions)
end




end
