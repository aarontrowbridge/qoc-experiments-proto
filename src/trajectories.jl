module Trajectories

using ..QubitSystems

export TrajectoryData

struct TrajectoryData
    states::Vector{Vector{Float64}}
    controls::Vector{Vector{Float64}}
    times::Vector{Float64}
end

# TODO: evaluate integrals of random control variable for augmented state

function TrajectoryData(
    system::AbstractQubitSystem{N},
    T::Int,
    Δt::Float64;
    σ=1.0
) where N

    states = [
        [system.ψ̃1; zeros(system.control_order + 1)],
        [σ * randn(system.nstates) for _ in 2:T-1]...,
        [system.ψ̃goal; zeros(system.control_order + 1)]
    ]

    controls = [
        zeros(N),
        [σ * randn(N) for _ in 2:T-1]...,
        zeros(N)
    ]

    times = [Δt * t for t = 0:T-1]

    return TrajectoryData(states, controls, times)
end




end
