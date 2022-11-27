using Pico

data_path = "experiments/ilc/g0_to_g1_T_581_dt_4.0_R_0.1_iter_3000ubound_0.0001_00001.jld2"
prob = load_problem(data_path)

println("Problem Loaded")

transmon_levels = 2
cavity_levels = 15


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

function g(x)
    abs2.(iso_to_ket(x))
end

gi(x) = x


function g_pop(x)
    y = []
    append!(y, sum(x[1:cavity_levels].^2 + x[2*cavity_levels .+ (1:cavity_levels)].^2))
    append!(y, sum(x[cavity_levels .+ (1:cavity_levels)].^2 + x[3*cavity_levels .+ (1:cavity_levels)].^2))
    for i = 1:5
        append!(y, x[i]^2 + x[i+2*cavity_levels]^2 + x[i + cavity_levels]^2 + x[i+3*cavity_levels]^2)
        #append!(y, x[i + cavity_levels]^2 + x[i+3*cavity_levels]^2)
    end
    return convert(typeof(x), y)
end

experiment = QuantumExperiment(
    prob.system,
    Ẑ.states[1],
    Ẑ.Δt,
    gi,
    60,
    1:Ẑ.T
)


max_iter = 20
fps = 5

prob = ILCProblem(
    prob.system,
    Ẑ,
    experiment;
    max_iter=max_iter,
    QP_verbose=false,
    QP_max_iter=100000,
    correction_term=true,
    norm_p=Inf,
    R=0.01,
    Q = 100.
)

solve!(prob)

# plot_dir = "plots/ILC/multimode"
# plot_path = generate_file_path("gif", data_name, plot_dir)

