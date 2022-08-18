using Pico
using JLD2

subprob_dir = "data/multimode/fixed_time/no_guess/problems"
experiment =
    "g0_to_g1_T_350_dt_1.0_R_0.001_iter_100_resolve_5_00000"

subprob_path = joinpath(subprob_dir, experiment * ".jld2")

subprob = load_prob(subprob_path)

# parameters

Rᵤ   = 1e1
Rₛ   = Rᵤ
iter = 100

mintime_options = Options(
    max_iter = iter,
)

mintime_prob = QuantumMinTimeProblem(
    subprob;
    Rᵤ=Rᵤ,
    Rₛ=Rₛ,
    mintime_options=mintime_options,
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
    subprob.system,
    mintime_prob.subprob.trajectory,
    plot_path
)

solve!(mintime_prob; save_path=save_path, solve_subprob=false)

plot_multimode(
    subprob.system,
    mintime_prob.subprob.trajectory,
    plot_path
)
