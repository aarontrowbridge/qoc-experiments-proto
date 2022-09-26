using Pico

data_name = "cnot_iter_500_00000"
data_dir = "data/twoqubit/cnot"
data_path = joinpath(data_dir, data_name * ".jld2")

prob = load_problem(data_path)

xs = [
    prob.trajectory.states[t][1:prob.system.isodim]
        for t = 1:prob.trajectory.T
]

us = [
    prob.trajectory.states[t][
        (prob.system.n_wfn_states +
        prob.system.∫a * prob.system.ncontrols) .+
        (1:prob.system.ncontrols)
    ] for t = 1:prob.trajectory.T
]

Ẑ = Trajectory(
    xs,
    us,
    prob.trajectory.times,
    prob.trajectory.T,
    prob.trajectory.Δt
)

g(x) = abs2.(iso_to_ket(x))

experiment = QuantumExperiment(
    prob.system,
    Ẑ.states[1],
    Ẑ.Δt,
    g,
    prob.system.isodim ÷ 2,
    1:Ẑ.T
)

prob = ILCProblem(
    prob.system,
    Ẑ,
    experiment;
    max_iter = 10
)

solve!(prob)
