using PicoOld

data_dir = "data/multimode/free_time/no_guess/problems"
prob_save_dir = "data/multimode/min_time/guess/problems"
controls_save_dir = "data/multimode/min_time/guess/controls"
plot_dir = "plots/multimode/min_time/guess"

data_name =
    "g0_to_g1_T_100_dt_4.0_Δt_max_factor_2.0_Q_1000.0_R_1.0e-5_iter_2000_u_bound_1.0e-5_alpha_transmon_20.0_alpha_cavity_20.0_resolve_3_00002"

data_path = joinpath(data_dir, data_name * ".jld2")

u_bound = 1e-5
u_bounds = fill(u_bound, 4)

options = Options(
    max_iter = 2000,
    max_cpu_time = 100_000_000.0,
)

mode = :min_time
equal_Δts = true

prob = load_problem(
    data_path;
    mode=mode,
    equal_Δts=equal_Δts,
    options=options,
    u_bounds=u_bounds,
)

experiment =
    "g0_to_g1_" *
    "T_$(prob.trajectory.T)_" *
    "Q_$(prob.params[:Q])_" *
    "R_$(prob.params[:R])_" *
    "u_bound_$(u_bound)_" *
    "iter_$(prob.params[:options].max_iter)"

plot_path = generate_file_path(
    "png",
    experiment,
    plot_dir
)

prob_save_path = generate_file_path(
    "jld2",
    experiment,
    prob_save_dir
)

controls_save_path = generate_file_path(
    "h5",
    experiment,
    controls_save_dir
)

plot_multimode_split(
    prob,
    plot_path;
    show_highest_modes=true
)

solve!(prob, save_path=prob_save_path)
save_controls(prob.trajectory, prob.system, controls_save_path)

plot_multimode_split(
    prob,
    plot_path;
    show_highest_modes=true
)
