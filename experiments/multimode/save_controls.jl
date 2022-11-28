using Pico

# experiment = "g0_to_g1_T_500_dt_0.8_R_1.0_iter_1000_resolve_9_00000_reload_iter_500_alpha_1.0_resolve_3_00000"

# experiment = "g0_to_g1_with_transmon_f_state_smooth_pulse_1_downsampled_5_T_101_dt_4.0_Q_500.0_R_1.0_alpha_transmon_20.0_alpha_cavity_20.0_iter_2000_resolve_10_00000"

# experiment = "g0_to_g1_T_102_dt_4.0_Q_200.0_R_0.1_u_bound_0.0001_iter_10000_00000"

# experiment = "g0_to_g1_T_101_dt_4.0_Q_500.0_R_0.1_u_bound_1.0e-5"

experiment = "g0_to_g1_T_102_dt_4.0_Q_200.0_R_0.1_u_bound_0.0001_iter_10000_00005"

data_dir = "data/multimode/min_time_update/guess/pinned/problems"
# data_dir = "data/multimode/fixed_time_update/guess/pinned/problems"

# save_dir = "data/multimode/good_solutions"
save_dir = "data/multimode/min_time_update/guess/pinned/controls"
# save_dir = "data/multimode/fixed_time_update/guess/pinned/controls"

data_path = joinpath(data_dir, experiment * ".jld2")

save_path = joinpath(save_dir, experiment * ".h5")

get_and_save_controls(data_path, save_path)
