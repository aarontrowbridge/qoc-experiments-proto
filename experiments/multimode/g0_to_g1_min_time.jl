using PicoOld

data_dir = "data/multimode/fixed_time/no_guess/problems"
prob_save_dir = "data/multimode/unitary/min_time/problems"
controls_save_dir = "data/multimode/unitary/min_time/controls"
plot_dir = "plots/multimode/unitary/min_time"

data_name = "g0_to_g1_transmon_4_cavity_14_T_200_dt_max_3.75_R_0.001_iter_3000_ubound_0.0001_00000"

data_path = joinpath(data_dir, data_name * ".jld2")

u_bound = 2e-4
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
    plot_path,
    4, 14;
    show_highest_modes=true
)

solve!(prob, save_path=prob_save_path)
save_controls(prob.trajectory, prob.system, controls_save_path)

plot_multimode_split(
    prob,
    plot_path,
    4, 14;
    show_highest_modes=true
)
