using Pico
using LinearAlgebra
using JLD2

transmon_levels = 2
cavity_levels = 14

ψ1 = "g0"
ψf = "g1"

system = MultiModeSystem(
    transmon_levels,
    cavity_levels,
    ψ1,
    ψf,
)

T                = 300
Δt               = 1.5
R                = 1.0
iter             = 10_000
resolves         = 10
pin_first_qstate = true
phase_flip       = false

# T                = parse(Int,     ARGS[1])
# Δt               = parse(Float64, ARGS[2])
# R                = parse(Float64, ARGS[3])
# iter             = parse(Int,     ARGS[4])
# resolves         = parse(Int,     ARGS[5])
# pin_first_qstate = parse(Bool,    ARGS[6])
# phase_flip       = parse(Bool,    ARGS[7])

options = Options(
    max_iter = iter,
    max_cpu_time = 100000.0,
)

u_bounds = BoundsConstraint(
    1:T,
    system.n_wfn_states .+
    slice(system.∫a + 1 + system.control_order, system.ncontrols),
    0.0001,
    system.vardim
)

cons = AbstractConstraint[u_bounds]

experiment = "$(ψ1)_to_$(ψf)_T_$(T)_dt_$(Δt)_R_$(R)_iter_$(iter)" * (pin_first_qstate ? "_pinned" : "") * (phase_flip ? "_phase_flip" : "") * "_mode_constrained"

plot_dir = "plots/multimode/fixed_time/no_guess"
data_dir = "data/multimode/fixed_time/no_guess/problems"

prob = QuantumControlProblem(
    system;
    T=T,
    Δt=Δt,
    R=R,
    pin_first_qstate=pin_first_qstate,
    options=options,
    constraints=cons
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
    plot_multimode(prob.system, prob.trajectory, plot_path)
    solve!(prob, save_path=save_path)
    plot_multimode(prob.system, prob.trajectory, plot_path)
    global prob = load_prob(save_path)
end
