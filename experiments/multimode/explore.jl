using Pico
using JLD2

data_dir = "data/multimode/fixed_time/no_guess/problems"

plot_dir = "plots/multimode/fixed_time/reloaded"

prob_name =
    "g0_to_g1_T_500_dt_0.8_R_1.0_iter_1000_resolve_10_00000"

data_path = joinpath(data_dir, prob_name*".jld2")
plot_path = joinpath(plot_dir, prob_name*"_highest_states_pre_reg.png")

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
