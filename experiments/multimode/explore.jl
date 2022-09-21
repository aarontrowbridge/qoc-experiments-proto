using Pico
using JLD2

data_dir = "data/multimode/fixed_time/guess/problems"

plot_dir = "plots/multimode/fixed_time/guess"

prob_name =
    "g0_to_g1_T_500_dt_0.8_R_1.0_iter_1000_resolve_9_00000_reload_iter_500_alpha_1.0_resolve_4_00000"

data_path = joinpath(data_dir, prob_name*".jld2")
plot_path = joinpath(plot_dir, prob_name*"_highest_states.png")

prob_data = load_data(data_path)


# @load data_path data

# save_trajectory(
#     data.trajectory,
#     joinpath(data_dir, prob_name*"_traj.h5")
# )

# traj = load_trajectory(
#     joinpath(data_dir, prob_name*"_traj.h5")
# )

plot_multimode_split(
    prob_data,
    plot_path;
    show_highest_modes=true,
)
