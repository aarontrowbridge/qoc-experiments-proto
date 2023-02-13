using PicoOld
using HDF5

data_dir = "data/multimode/fixed_time/guess/controls"
save_dir = "data/multimode/good_solutions"

data_name = "g0_to_g1_with_transmon_f_state_smooth_pulse_1_downsampled_5_T_101_dt_4.0_R_1.0_alpha_transmon_20.0_alpha_cavity_20.0_iter_2000_resolve_6_00000"

data_path = joinpath(data_dir, data_name*".h5")

h5open(data_path, "r") do f
    global controls = f["controls"][:,:] |> transpose
    global Δt = f["delta_t"][]
end

transmon_levels = 3
cavity_levels = 14
ψ1 = "g0"
ψf = "g1"
u_bound = 1e-5

system = MultiModeSystem(
    transmon_levels,
    cavity_levels,
    ψ1,
    ψf
)

traj = Trajectory(
    system,
    controls,
    Δt
)

Q             = 500.0
R             = 1.0e-1
iter          = 2000
resolves      = 5
α_transmon    = 20.0
α_cavity      = 20.0

options = Options(
    max_iter = iter,
    max_cpu_time = 100000.0,
)

experiment =
    "$(ψ1)_to_$(ψf)_T_$(traj.T)_dt_$(traj.Δt)_Q_$(Q)_R_$(R)" *
    "_u_bound_$(u_bound)"

plot_dir = "plots/multimode/fixed_time_update/guess/good_solutions"
data_dir = "data/multimode/fixed_time_update/guess/good_solutions/problems"

ketdim = transmon_levels * cavity_levels

highest_cavity_modes = [
    cavity_levels .* [1, 2];
    ketdim .+ (cavity_levels .* [1, 2])
]

transmon_f_states = [
    2 * cavity_levels .+ [1:cavity_levels...];
    ketdim .+ (2 * cavity_levels .+ [1:cavity_levels...])
]

reg_states = [highest_cavity_modes; transmon_f_states]

prob = QuantumControlProblem(
    system,
    traj;
    R=R,
    options=options,
    u_bounds=fill(u_bound, system.ncontrols),
    L1_regularized_states=reg_states,
    α=[
        fill(α_transmon, length(transmon_f_states));
        fill(α_cavity, length(highest_cavity_modes))
    ],
)

save_name = "g0_to_g1_T_101_dt_4.0_R_1.0_alpha_transmon_20.0_alpha_cavity_20.0_iter_2000_resolve_6_00000"

save_problem(prob, joinpath(save_dir, experiment*".jld2"))

plot_path = joinpath(save_dir, experiment*".png")

plot_multimode_split(
    prob,
    plot_path;
    show_highest_modes=true
)



# for i = 1:resolves

#     resolve = "_resolve_$i"

#     plot_path = generate_file_path(
#         "png",
#         experiment * resolve,
#         plot_dir
#     )

#     save_path = generate_file_path(
#         "jld2",
#         experiment * resolve,
#         data_dir
#     )

#     plot_multimode_split(
#         prob,
#         plot_path;
#         show_highest_modes=true
#     )

#     solve!(prob, save_path=save_path)

#     plot_multimode_split(
#         prob,
#         plot_path;
#         show_highest_modes=true
#     )

#     global prob = load_problem(save_path)
# end
