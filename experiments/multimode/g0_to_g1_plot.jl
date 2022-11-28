using Pico

data_dir = "data/multimode/good_solutions"

data_name = "g0_to_g1_T_101_dt_4.0_Q_500.0_R_0.1_u_bound_1.0e-5"

data_path = joinpath(data_dir, data_name * ".jld2")

data = load_data(data_path)

# plot Trajectory

plot_name = data_name * "_full_trajectory"

plot_path = joinpath(data_dir, plot_name * ".png")

plot_multimode_split(
    data,
    plot_path;
    show_highest_modes=false
)
