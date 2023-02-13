using PicoOld

experiment = "cnot_iter_200_00000"

data_dir  = "data/twoqubit/cnot_update"
data_path = joinpath(data_dir, experiment * ".jld2")

mode = :min_time
R = 1e-7
max_iter = 300
u_bound = 1e1

experiment = experiment * "_mintime_R_$(R)_u_bound_$(u_bound)_iter_$(max_iter)"

save_dir = "data/twoqubit/cnot_update/min_time"
save_path = generate_file_path("jld2", experiment, save_dir)

plot_dir = "plots/twoqubit/cnot_update/min_time"
plot_path = generate_file_path("png", experiment, plot_dir)

prob = load_problem(
    data_path;
    mode=mode,
    R=R,
    u_bounds=fill(u_bound, 4),
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
