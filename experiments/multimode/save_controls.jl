using Pico

data_dir = "data/multimode/min_time/no_guess/problems"
experiment = "g0_to_g1_T_500_dt_0.8_R_1.0_iter_1000_resolve_10_00000_Ru_10.0_mintime_iter_10000_resolve_6"

data_path = joinpath(data_dir, experiment*".jld2")

save_dir = "data/multimode/min_time/no_guess/controls"
save_path = joinpath(save_dir, experiment*".h5")

get_and_save_controls(data_path, save_path)
