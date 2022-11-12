using Pico

experiment = "cnot_iter_5000_00000"

data_dir  = "data/twoqubit/cnot_update"
save_dir = "data/twoqubit/cnot_update/min_time"
plot_dir = "plots/twoqubit/cnot_update/min_time"

data_path = joinpath(data_dir, experiment * ".jld2")
save_path = generate_file_path("jld2", experiment, save_dir)
plot_path = generate_file_path("png", experiment, plot_dir)

mode = :min_time
R = 1e-7
max_iter = 300

prob = load_problem(
    data_path;
    mode=mode,
    R=R,
    options=Options(
        max_iter=max_iter,
    )
)

plot_twoqubit(
    prob.system,
    prob.trajectory,
    plot_path;
    fig_title = "CNOT gate",
    i = 4
)

solve!(prob; save_path=save_path)

plot_twoqubit(
    prob.system,
    prob.trajectory,
    plot_path;
    fig_title = "CNOT gate",
    i = 4
)
