using Pico
using JLD2

data_dir = "data/multimode/fixed_time/no_guess/problems"

plot_dir = "plots/multimode/fixed_time/no_guess"

prob_name =
    "g0_to_g1_T_300_dt_1.5_R_1.0_iter_1000_resolve_10_00000"

data_path = joinpath(data_dir, prob_name*".jld2")
plot_path = joinpath(plot_dir, prob_name*"_highest_states.png")

@load data_path data

save_trajectory(
    data.trajectory,
    joinpath(data_dir, prob_name*"_traj.h5")
)

traj = load_trajectory(
    joinpath(data_dir, prob_name*"_traj.h5")
)

plot_multimode(
    data.system,
    traj,
    plot_path;
    components=[14, 28]
)
