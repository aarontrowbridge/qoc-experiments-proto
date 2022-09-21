using Pico
using JLD2

data_dir = "data/multimode/fixed_time/no_guess/problems"
experiment = "g0_to_g1_T_500_dt_0.8_R_1.0_iter_20_pinned_alpha_0.25_resolve_10_00000"

data_path = joinpath(data_dir, experiment*".jld2")

@load data_path data

iter = 1000

options = Options(
    max_iter = iter,
    max_cpu_time = 10_000_000.0,
)

data.params[:options] = options

plot_dir = "plots/multimode/fixed_time/guess"

prob = QuantumControlProblem(data)

resolves = 10

for i = 1:resolves
    resolve = "_reload_iter_$(iter)_resolve_$i"
    plot_path = generate_file_path(
        "png",
        experiment * resolve,
        plot_dir
    )
    save_path = generate_file_path(
        "jld2",
        experiment * resolve,
        data_dir
    )
    plot_multimode_split(prob, plot_path)
    solve!(prob, save_path=save_path)
    plot_multimode_split(prob, plot_path)
    global prob = load_problem(save_path)
end
