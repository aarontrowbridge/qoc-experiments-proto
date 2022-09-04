using Pico
using LinearAlgebra
using JLD2


const EXPERIMENT_NAME = "g0_to_g1"
plot_path = generate_file_path("png", EXPERIMENT_NAME * "_iter_$(iter)", "plots/multimode/fermium/")

const TRANSMON_LEVELS = 2
const CAVITY_LEVELS = 14

function cavity_state(level)
    state = zeros(CAVITY_LEVELS)
    state[level + 1] = 1.
    return state
end
#const TRANSMON_ID = I(TRANSMON_LEVELS)

TRANSMON_G = [1; zeros(TRANSMON_LEVELS - 1)]
TRANSMON_E = [zeros(1); 1; zeros(TRANSMON_LEVELS - 2)]


CHI = 2π * -0.5469e-3
KAPPA = 2π * 4e-6

H_drift = 2 * CHI * kron(TRANSMON_E*TRANSMON_E', number(CAVITY_LEVELS)) +
          (KAPPA/2) * kron(I(TRANSMON_LEVELS), quad(CAVITY_LEVELS))

transmon_driveR = kron(create(TRANSMON_LEVELS) + annihilate(TRANSMON_LEVELS), I(CAVITY_LEVELS))
transmon_driveI = kron(1im*(create(TRANSMON_LEVELS) - annihilate(TRANSMON_LEVELS)), I(CAVITY_LEVELS))

cavity_driveR = kron(I(TRANSMON_LEVELS), create(CAVITY_LEVELS) + annihilate(CAVITY_LEVELS))
cavity_driveI = kron(I(TRANSMON_LEVELS),  1im * (create(CAVITY_LEVELS) - annihilate(CAVITY_LEVELS)))

H_drives = [transmon_driveR, transmon_driveI, cavity_driveR, cavity_driveI]

ψ1 = kron(TRANSMON_G, cavity_state(0))
ψf = kron(TRANSMON_G, cavity_state(1))

# bounds on controls

qubit_a_bounds = [0.018 * 2π, 0.018 * 2π]

cavity_a_bounds = fill(0.03, length(H_drives) - 2)

a_bounds = [qubit_a_bounds; cavity_a_bounds]

system = QuantumSystem(
    H_drift,
    H_drives,
    ψ1,
    ψf,
    a_bounds
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
    max_cpu_time = 30000.0,
    tol = 1e-6
)

u_bounds = BoundsConstraint(
    1:T,
    system.n_wfn_states .+
    slice(system.∫a + 1 + system.control_order, system.ncontrols),
    0.0001,
    system.vardim
)

energy_con = EqualityConstraint(
    2:T-1,
    [CAVITY_LEVELS, 2 * CAVITY_LEVELS, 3 * CAVITY_LEVELS, 4 * CAVITY_LEVELS],
    0.0,
    system.vardim;
    name="highest energy level constraints"
)

cons = AbstractConstraint[u_bounds, energy_con]

experiment = "g0_to_g1_T_$(T)_dt_$(Δt)_R_$(R)_iter_$(iter)" * (pin_first_qstate ? "_pinned" : "") * (phase_flip ? "_phase_flip" : "") * "_mode_constrained"

plot_dir = "plots/multimode/fermium"
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
