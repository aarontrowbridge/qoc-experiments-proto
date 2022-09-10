using Pico
using JLD2

data_dir = "data/multimode/fixed_time/no_guess/problems"
experiment = "g0_to_g1_T_500_dt_0.8_R_1.0_iter_1000_resolve_10_00000"

data_path = joinpath(data_dir, experiment*".jld2")

@load data_path data

R        = 1.0
iter     = 500
resolves = 10
αval     = 0.1

u_bounds = BoundsConstraint(
    1:data.trajectory.T,
    data.system.n_wfn_states .+
    slice(
        data.system.∫a + 1 + data.system.control_order,
        data.system.ncontrols
    ),
    0.0001,
    data.system.vardim
)

cons = AbstractConstraint[u_bounds]

plot_dir = "plots/multimode/fixed_time/reloaded"
data_dir = "data/multimode/fixed_time/no_guess/problems"

options = Options(
    max_iter = iter,
    max_cpu_time = 100000.0,
)

prob = QuantumControlProblem(
    data.system,
    data.trajectory;
    R=R,
    options=options,
    constraints=cons,
    L1_regularized_states=14 .* [1, 2, 3, 4],
    α=fill(αval, 4)
)

info = "_L1_alpha_$(αval)"

for i = 1:resolves
    resolve = "_resolve_$i"
    plot_path = generate_file_path(
        "png",
        experiment * info * resolve,
        plot_dir
    )
    save_path = generate_file_path(
        "jld2",
        experiment * info * resolve,
        data_dir
    )
    plot_multimode_split(prob, plot_path)
    solve!(prob, save_path=save_path)
    plot_multimode_split(prob, plot_path)
    global prob = load_problem(save_path)
end
