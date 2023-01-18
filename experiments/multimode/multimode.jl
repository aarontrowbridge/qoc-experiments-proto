using Pico

transmon_levels = 3
cavity_levels = 14

ψ1 = "g0"
ψf = "g1"

T             = 100
Δt            = 10.0
Q             = 1000.0
R             = 1.0e-5
iter          = 2000
resolves      = 10
α_transmon    = 20.0
α_cavity      = 20.0
u_bound       = 1e-5
Δt_max_factor = 1.0

system = MultiModeSystem(
    transmon_levels,
    cavity_levels,
    ψ1,
    ψf
)

options = Options(
    max_iter = iter,
    max_cpu_time = 100000.0,
)

experiment =
    "$(ψ1)_to_$(ψf)_T_$(T)_dt_$(Δt)_dt_max_factor_$(Δt_max_factor)_Q_$(Q)_R_$(R)_iter_$(iter)" *
    "_u_bound_$(u_bound)" *
    "_alpha_transmon_$(α_transmon)_alpha_cavity_$(α_cavity)"

plot_dir = "plots/multimode/free_time/no_guess"
data_dir = "data/multimode/free_time/no_guess/problems"
controls_dir = "data/multimode/free_time/no_guess/controls"

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
    system;
    T=T,
    Δt=Δt,
    Δt_max=Δt_max_factor * Δt,
    R=R,
    options=options,
    L1_regularized_states=reg_states,
    α=[
        fill(α_transmon, length(transmon_f_states));
        fill(α_cavity, length(highest_cavity_modes))
    ],
    u_bounds=fill(u_bound, 4)
)

for i = 1:resolves

    resolve = "_resolve_$i"

    plot_path = generate_file_path(
        "png",
        experiment * resolve,
        plot_dir
    )

    prob_save_path = generate_file_path(
        "jld2",
        experiment * resolve,
        data_dir
    )

    controls_save_path = generate_file_path(
        "h5",
        experiment * resolve,
        controls_dir
    )

    plot_multimode_split(
        prob,
        plot_path;
        show_highest_modes=true
    )

    solve!(prob, save_path=prob_save_path)

    save_controls(prob.trajectory, prob.system, controls_save_path)

    plot_multimode_split(
        prob,
        plot_path;
        show_highest_modes=true
    )

    global prob = load_problem(prob_save_path)
end
