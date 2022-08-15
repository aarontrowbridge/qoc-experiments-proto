using Pico
using LinearAlgebra
using JLD2

iter = 4000

const EXPERIMENT_NAME = "g0_to_g1"
# plot_path = generate_file_path("png", EXPERIMENT_NAME * "_iter_$(iter)", "plots/multimode/rewrite/")

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

cavity_a_bounds = fill(0.03, sys.ncontrols - 2)

a_bounds = [qubit_a_bounds; cavity_a_bounds]

system = QuantumSystem(
    H_drift,
    H_drives,
    ψ1,
    ψf,
    control_bounds = a_bounds
)

T = parse(Int, ARGS[1])
Δt = parse(Float64, ARGS[2])
R = parse(Float64, ARGS[3])
iter = parse(Int, ARGS[4])
pin_first_qstate = parse(Bool, ARGS[5])
phase_flip = parse(Bool, ARGS[6])

options = Options(
    max_iter = iter,
    max_cpu_time = 100000.0,
)



u_bounds = BoundsConstraint(
    1:T,
    sys.n_wfn_states .+
    slice(sys.∫a + 1 + sys.control_order, sys.ncontrols),
    0.0001,
    sys.vardim
)

cons = AbstractConstraint[u_bounds]

experiment = "g0_to_g1_T_$(T)_dt_$(Δt)_R_$(R)_iter_$(iter)" * (pin_first_qstate ? "_pinned" : "") * (phase_flip ? "_phase_flip" : "")

plot_dir = "plots/multimode/fixed_time/no_guess"
data_dir = "data/multimode/fixed_time/no_guess/problems"

resolves = parse(Int, ARGS[end])

prob = QuantumControlProblem(
    system,
    T;
    Δt=Δt,
    R=R,
    a_bound=a_bounds,
    phase_flip=phase_flip,
    pin_first_qstate=pin_first_qstate,
    options=options,
    cons=cons
)

for i = 1:resolves
    resolve = "_resolve_$i"
    plot_path = generate_file_path(
        "png",
        experiment * resolve,
        plot_dir
    )
    data_path = generate_file_path(
        "jld2",
        experiment * resolve,
        data_dir
    )
    plot_multimode(sys, prob.trajectory, plot_path)
    solve!(prob, save=true, path=data_path)
    plot_multimode(sys, prob.trajectory, plot_path)
    global prob = load_object(data_path)
end
