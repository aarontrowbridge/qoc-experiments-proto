using Pico

data_dir = "data/multimode/good_solutions"
save_dir = "data/multimode/min_time_update/guess/good_solutions/problems"
plot_dir = "plots/multimode/min_time_update/guess/good_solutions/pinned"

# data_name = "g0_to_g1_T_102_dt_4.0_Q_200.0_R_0.1_u_bound_0.0001_iter_10000_00000"
# data_name = "g0_to_g1_T_102_dt_4.0_Q_200.0_R_0.1_u_bound_0.0001_iter_10000_00004"
data_name = "g0_to_g1_T_101_dt_4.0_Q_500.0_R_0.1_u_bound_1.0e-5"

data_path = joinpath(data_dir, data_name * ".jld2")

u_bound = 1e-4
u_bounds = fill(u_bound, 4)

options = Options(
    max_iter = 300,
    max_cpu_time = 100_000_000.0,
)

mode = :min_time
equal_Δts = false

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
