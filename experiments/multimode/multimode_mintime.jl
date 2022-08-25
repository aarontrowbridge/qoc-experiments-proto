using Pico
using JLD2

subprob_data_dir = "data/multimode/fixed_time/no_guess/problems"

experiment =
    "g0_to_g1_T_350_dt_1.5_R_1.0_iter_1000_resolve_10_00000"

subprob_data_path =
    joinpath(subprob_data_dir, experiment * ".jld2")

@load subprob_data_path data

# parameters

Rᵤ   = 1e1
Rₛ   = Rᵤ
iter = 1000

prob = QuantumMinTimeProblem(
    data;
    Rᵤ=Rᵤ,
    Rₛ=Rₛ,
    mintime_options=Options(
        max_iter=iter,
    )
)

mintime_info = "_Ru_$(Rᵤ)_mintime_iter_$(iter)"

save_dir = "data/multimode/min_time/no_guess/problems"
save_path = joinpath(
    save_dir,
    experiment * mintime_info * ".jld2"
)

plot_dir = "plots/multimode/min_time/no_guess"
plot_path = joinpath(
    plot_dir,
    experiment * mintime_info * ".png"
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
