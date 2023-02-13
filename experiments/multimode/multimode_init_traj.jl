using PicoOld
using JLD2

data_dir = "data/multimode/fixed_time/guess/problems"
experiment = "g0_to_g1_with_transmon_f_state_smooth_pulse_1_downsampled_5_T_101_dt_4.0_R_1.0_alpha_transmon_20.0_alpha_cavity_20.0_iter_2000_resolve_6_00000"

data_path = joinpath(data_dir, experiment*".jld2")

# prob = load_prob(data_path)

@load data_path data

# R        = 1.0
# iter     = 500
# resolves = 10
# αval     = 2.0
# mode_con = true

# u_bounds = BoundsConstraint(
#     1:data.trajectory.T,
#     data.system.n_wfn_states .+
#     slice(
#         data.system.∫a + 1 + data.system.control_order,
#         data.system.ncontrols
#     ),
#     0.0001,
#     data.system.vardim
# )

# cons = AbstractConstraint[u_bounds]

# plot_dir = "plots/multimode/fixed_time/guess"
# data_dir = "data/multimode/fixed_time/guess/problems"

# options = Options(
#     max_iter = iter,
#     max_cpu_time = 100000.0,
# )

# if mode_con
#     prob = QuantumControlProblem(
#         data.system,
#         data.trajectory;
#         R=R,
#         options=options,
#         constraints=cons,
#         L1_regularized_states=14 .* [1, 2, 3, 4],
#         α=fill(αval, 4)
#     )
#     info = "_reload_iter_$(iter)_alpha_$(αval)"
# else
#     prob = QuantumControlProblem(
#         data.system,
#         data.trajectory;
#         R=R,
#         options=options,
#         constraints=cons
#     )
#     info = "_reload_iter_$(iter)"
# end


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
