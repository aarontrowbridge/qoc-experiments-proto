using Pico

data_dir = "data/multimode/free_time/no_guess/problems"

# experiment =
    # "g0_to_g1_T_500_dt_0.8_R_1.0_iter_1000_resolve_10_00000"

experiment =
    "g0_to_g1_T_100_dt_4.0_Δt_max_factor_2.0_Q_1000.0_R_1.0e-5_iter_2000_u_bound_1.0e-5_alpha_transmon_20.0_alpha_cavity_20.0_resolve_1_00000"

subprob_data_path =
    joinpath(subprob_data_dir, experiment * ".jld2")

subprob_data = load_data(subprob_data_path)

# parameters

Rᵤ   = 1e2
Rₛ   = Rᵤ
iter = 10000

resolves = 1

prob = MinTimeQuantumControlProblem(;
    subprob_data=subprob_data,
    Rᵤ=Rᵤ,
    Rₛ=Rₛ,
    mintime_options=Options(
        max_iter=iter,
    )
)

mintime_info = "_Ru_$(Rᵤ)_mintime_iter_$(iter)"

prob_save_dir = "data/multimode/min_time/no_guess/problems"
controls_save_dir = "data/multimode/min_time/no_guess/controls"

plot_dir = "plots/multimode/min_time/guess"

# save_path = joinpath(
#     save_dir,
#     experiment * mintime_info * ".jld2"
# )

# plot_path = joinpath(
#     plot_dir,
#     experiment * mintime_info * ".png"
# )

# plot_multimode(
#     prob.subprob.system,
#     prob.subprob.trajectory,
#     plot_path
# )

# solve!(prob; save_path=save_path, solve_subprob=false)

# plot_multimode(
#     prob.subprob.system,
#     prob.subprob.trajectory,
#     plot_path
# )

# resolve problem

for i = 1:resolves
    resolve = "_resolve_$i"

    save_path = joinpath(
        save_dir,
        experiment * mintime_info * resolve * ".jld2"
    )

    plot_path = joinpath(
        plot_dir,
        experiment * mintime_info * resolve * ".png"
    )

    plot_multimode_split(
        prob.subprob,
        plot_path;
        show_highest_modes=true
    )

    solve!(prob; save_path=prob_save_path, solve_subprob=false)
    save_controls(prob.trajectory, prob.system, controls_save_path)

    plot_multimode_split(
        prob.subprob,
        plot_path;
        show_highest_modes=true
    )

    prob_data = load_data(prob_save_path)

    global prob = MinTimeQuantumControlProblem(;
        prob_data=prob_data,
        Rᵤ=Rᵤ,
        Rₛ=Rₛ,
        mintime_options=Options(
            max_iter=iter,
        )
    )
end
