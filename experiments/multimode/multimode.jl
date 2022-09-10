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

T                = 500
Δt               = 0.8
R                = 1.0
iter             = 100
resolves         = 10
pin_first_qstate = true
phase_flip       = false
mode_con         = true
αval             = 0.25

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

experiment = "$(ψ1)_to_$(ψf)_T_$(T)_dt_$(Δt)_R_$(R)_iter_$(iter)" * (pin_first_qstate ? "_pinned" : "") * (phase_flip ? "_phase_flip" : "") * (mode_con ? "_mode_constrained_alpha_$(αval)" : "")

plot_dir = "plots/multimode/fixed_time/no_guess"
data_dir = "data/multimode/fixed_time/no_guess/problems"

if mode_con
    prob = QuantumControlProblem(
        system;
        T=T,
        Δt=Δt,
        R=R,
        pin_first_qstate=pin_first_qstate,
        options=options,
        constraints=cons,
        L1_regularized_states=[1,2,3,4] .* cavity_levels,
        α = fill(αval, 4),
    )
else
    prob = QuantumControlProblem(
        system;
        T=T,
        Δt=Δt,
        R=R,
        pin_first_qstate=pin_first_qstate,
        options=options,
        constraints=cons
    )
end



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
    plot_multimode_split(prob, plot_path)
    solve!(prob, save_path=save_path)
    plot_multimode_split(prob, plot_path)
    global prob = load_problem(save_path)
end
