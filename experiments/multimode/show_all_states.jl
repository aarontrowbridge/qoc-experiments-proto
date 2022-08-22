using Pico

prob_dir = "data/multimode/min_time/no_guess/problems"
plot_dir = "plots/multimode/min_time/no_guess"

prob_name = "g0_to_g1_T_350_dt_1.0_R_0.001_iter_100_resolve_5_00000_Ru_10.0_mintime_iter_100"

prob_path = joinpath(prob_dir, prob_name*".jld2")
plot_path = joinpath(plot_dir, prob_name*"_allstates.png")

prob = load_prob(prob_path)

plot_multimode(
    prob.system,
    prob.trajectory,
    plot_path
)
