using Pico
using JLD2

subprob_data_dir = "data/multimode/fixed_time/no_guess/problems"

experiment =
    "g0_to_g1_T_500_dt_0.8_R_1.0_iter_1000_resolve_10_00000"

subprob_data_path =
    joinpath(subprob_data_dir, experiment * ".jld2")

subprob_data = load_data(subprob_data_path)

# parameters

Rᵤ   = 1e1
Rₛ   = Rᵤ
iter = 10000

resolves = 10

prob = MinTimeQuantumControlProblem(
    subprob_data;
    Rᵤ=Rᵤ,
    Rₛ=Rₛ,
    mintime_options=Options(
        max_iter=iter,
    )
)

mintime_info = "_Ru_$(Rᵤ)_mintime_iter_$(iter)"

save_dir = "data/multimode/min_time/no_guess/problems"

plot_dir = "plots/multimode/min_time/no_guess"

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

    plot_multimode(
        prob.subprob.system,
        prob.subprob.trajectory,
        plot_path
    )

    solve!(prob; save_path=save_path, solve_subprob=false)

    plot_multimode(
        prob.subprob.system,
        prob.subprob.trajectory,
        plot_path
    )

    prob_data = load_data(save_path)

    global prob = MinTimeQuantumControlProblem(
        prob_data;
        Rᵤ=Rᵤ,
        Rₛ=Rₛ,
        mintime_options=Options(
            max_iter=iter,
        )
    )
end
