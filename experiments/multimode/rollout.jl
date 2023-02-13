using PicoOld
using JLD2

data_dir = "data/multimode/fixed_time/no_guess/problems"

plot_dir = "plots/multimode/fixed_time/no_guess"

prob_name =
    "g0_to_g1_T_300_dt_1.5_R_1.0_iter_1000_resolve_10_00000"

data_path = joinpath(data_dir, prob_name*".jld2")
plot_path = joinpath(plot_dir, prob_name*"_rollout.png")

@load data_path data

controls = controls_matrix(data.trajectory, data.system)

rollout_traj = Trajectory(
    data.system,
    controls,
    data.trajectory.Δt
)

plot_multimode(
    data.system,
    rollout_traj,
    plot_path;
    # components=[14, 28]
)
