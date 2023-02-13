using PicoOld
using LinearAlgebra

data_name = "cnot_iter_200_00000_mintime_R_1.0e-7_u_bound_10.0_iter_300_00000"
data_dir = "data/twoqubit/cnot_update/min_time/"
data_path = joinpath(data_dir, data_name * ".jld2")

prob = load_data(data_path)

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

g(ψ̃) = abs2.(iso_to_ket(ψ̃))

# G_error_term = 0.025 * Pico.QuantumSystems.iso(kron(GATES[:X], GATES[:Y]))
G_error_term = 0.025 * Pico.QuantumSystems.iso(GATES[:CX])

experiment = QuantumExperiment(
    prob.system,
    Ẑ.states[1],
    Ẑ.times,
    x -> x,
    prob.system.isodim,
    1:Ẑ.T;
    integrator=exp,
    G_error_term=G_error_term
)


max_iter = 25
fps = 2
R = 0.1
Q = 0.0

prob = ILCProblem(
    prob.system,
    Ẑ,
    experiment;
    max_iter=max_iter,
    QP_verbose=false,
    QP_max_iter=100000,
    correction_term=false,
    norm_p=2,
    Q=Q,
    R=R,
)

solve!(prob)

plot_dir = "plots/ILC/twoqubit_update"
plot_path = generate_file_path("gif", data_name, plot_dir)

animate_ILC(prob, plot_path; fps=fps)
