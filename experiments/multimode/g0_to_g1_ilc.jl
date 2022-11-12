using Pico

data_dir = "data/multimode/fixed_time_update/guess/problems"

data_name = "g0_to_g1_T_102_dt_4.0_Q_500.0_R_0.1_iter_2000_u_bound_1.0e-5_alpha_transmon_20.0_alpha_cavity_20.0_resolve_5_00000"

data_path = joinpath(data_dir, data_name * ".jld2")

data = load_data(data_path)

xs = [
    data.trajectory.states[t][1:data.system.isodim]
        for t = 1:data.trajectory.T
]

us = [
    data.trajectory.states[t][
        (data.system.n_wfn_states +
        data.system.∫a * data.system.ncontrols) .+
        (1:data.system.ncontrols)
    ] for t = 1:data.trajectory.T
]

Ẑ = Trajectory(
    xs,
    us,
    data.trajectory.times,
    data.trajectory.T,
    data.trajectory.Δt
)

transmon_levels = 3
cavity_levels = 14
ψ1 = "g0"
ψf = "g1"
χ = 1.0 * 2π * -0.5459e-3


experimental_system = MultiModeSystem(
    transmon_levels,
    cavity_levels,
    ψ1,
    ψf;
    χ=χ
)

g(ψ̃) = abs2.(iso_to_ket(ψ̃))

experiment = QuantumExperiment(
    experimental_system,
    Ẑ.states[1],
    Ẑ.Δt,
    x -> x,
    data.system.isodim,
    1:Ẑ.T
)

max_iter = 20
fps = 5

prob = ILCProblem(
    data.system,
    Ẑ,
    experiment;
    max_iter=max_iter,
    QP_verbose=false,
    QP_max_iter=100000,
    correction_term=true,
    norm_p=Inf,
    R=0.01,
    Q=0.0
)

solve!(prob)
