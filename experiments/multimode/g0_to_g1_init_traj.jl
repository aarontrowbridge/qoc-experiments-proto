using PicoOld

# data_dir = "data/multimode/fixed_time_update/guess/pinned/problems"

data_dir = "data/multimode/good_solutions"
save_dir = "data/multimode/fixed_time_update/guess/pinned/problems"
plot_dir = "plots/multimode/fixed_time_update/guess/pinned"

# data_name = "g0_to_g1_T_102_dt_4.0_Q_500.0_R_0.1_iter_2000_u_bound_1.0e-5_alpha_transmon_20.0_alpha_cavity_20.0_resolve_5_00000_00001_00000"
data_name = "g0_to_g1_T_101_dt_4.0_Q_500.0_R_0.1_u_bound_1.0e-5"

data_path = joinpath(data_dir, data_name * ".jld2")

pin_first_qstate = true

u_bound = 1e-4
u_bounds = fill(u_bound, 4)

options = Options(
    max_iter = 500,
    max_cpu_time = 100_000.0,
)


prob = load_problem(
    data_path;
    pin_first_qstate=pin_first_qstate,
    options=options,
    u_bounds=u_bounds,
    Δt_max=4.0
)

experiment =
    "g0_to_g1_" *
    "T_$(prob.trajectory.T)_" *
    "dt_$(prob.trajectory.Δt)_" *
    "Q_$(prob.params[:Q])_" *
    "R_$(prob.params[:R])_" *
    "u_bound_$(u_bound)_" *
    "iter_$(prob.params[:options].max_iter)"

plot_path = generate_file_path(
    "png",
    experiment,
    plot_dir
)

save_path = generate_file_path(
    "jld2",
    experiment,
    save_dir
)

plot_multimode_split(
    prob,
    plot_path;
    show_highest_modes=true
)

solve!(prob, save_path=save_path)

plot_multimode_split(
    prob,
    plot_path;
    show_highest_modes=true
)
