using PicoOld
using JLD2

data_dir = "data/multimode/fixed_time/guess/problems"
data_name = "g0_to_g1_T_500_dt_0.8_R_1.0_iter_1000_resolve_9_00000_reload_iter_500_alpha_1.0_resolve_3_00000"

data_path = joinpath(data_dir, data_name*".jld2")

data = load_data(data_path)

Q          = 500.0
R          = 1.0
iter       = 2000
resolves   = 10
downsamp   = 5

α_transmon = 20.0
α_cavity   = 20.0




# system = data.system

transmon_levels = 3
cavity_levels = 14

ψ1 = "g0"
ψf = "g1"

system = MultiModeSystem(
    transmon_levels,
    cavity_levels,
    ψ1,
    ψf
)

down_traj = dumb_downsample(data.trajectory, downsamp)
init_controls = controls_matrix(down_traj, data.system)

init_traj = Trajectory(system, init_controls, down_traj.Δt)


experiment = "g0_to_g1_with_transmon_f_state_smooth_pulse_1_downsampled_$(downsamp)_T_$(init_traj.T)_dt_$(init_traj.Δt)_Q_$(Q)_R_$(R)_alpha_transmon_$(α_transmon)_alpha_cavity_$(α_cavity)_iter_$(iter)"

u_bounds = BoundsConstraint(
    1:init_traj.T,
    system.n_wfn_states .+
    slice(
        system.∫a + 1 + system.control_order,
        system.ncontrols
    ),
    0.0001,
    system.vardim
)

cons = AbstractConstraint[u_bounds]

plot_dir = "plots/multimode/fixed_time/guess"
data_dir = "data/multimode/fixed_time/guess/problems"

options = Options(
    max_iter = iter,
    max_cpu_time = 100000.0,
)

ketdim = transmon_levels * cavity_levels

highest_cavity_modes =
    [
        cavity_levels .* [1, 2];
        ketdim .+ (cavity_levels .* [1, 2])
    ]

transmon_f_states =
    [
        2 * cavity_levels .+ [1:cavity_levels...];
        ketdim .+ (2 * cavity_levels .+ [1:cavity_levels...])
    ]

reg_states = [highest_cavity_modes; transmon_f_states]

prob = QuantumControlProblem(
    system,
    init_traj;
    Q=Q,
    R=R,
    options=options,
    constraints=cons,
    L1_regularized_states=reg_states,
    α=[
        fill(α_transmon, length(transmon_f_states));
        fill(α_cavity, length(highest_cavity_modes))
    ],
)

for i = 1:resolves
    resolve = "_resolve_$i"
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
    plot_multimode_split(prob, plot_path; show_highest_modes=true)
    solve!(prob, save_path=save_path)
    plot_multimode_split(prob, plot_path; show_highest_modes=true)
    global prob = load_problem(save_path)
end
