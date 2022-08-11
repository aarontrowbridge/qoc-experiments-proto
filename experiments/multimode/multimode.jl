using QubitControl
using HDF5
using LinearAlgebra

iter = 4000

const EXPERIMENT_NAME = "g0_to_g1"
plot_path = generate_file_path("png", EXPERIMENT_NAME * "_iter_$(iter)", "plots/multimode/rewrite/")

const TRANSMON_LEVELS = 2 
const CAVITY_LEVELS = 12

function cavity_state(level)
    state = zeros(CAVITY_LEVELS)
    state[level + 1] = 1.
    return state
end
#const TRANSMON_ID = I(TRANSMON_LEVELS)

const TRANSMON_G = [1; zeros(TRANSMON_LEVELS - 1)]
const TRANSMON_E = [zeros(1); 1; zeros(TRANSMON_LEVELS - 2)]


const CHI = 2π * -0.5469e-3
const KAPPA = 2π * 4e-6

H_drift = 2 * CHI * kron(TRANSMON_E*TRANSMON_E', number(CAVITY_LEVELS)) +
          (KAPPA/2) * kron(I(TRANSMON_LEVELS), quad(CAVITY_LEVELS))

transmon_driveR = kron(create(TRANSMON_LEVELS) + annihilate(TRANSMON_LEVELS), I(CAVITY_LEVELS))
transmon_driveI = kron(1im*(create(TRANSMON_LEVELS) - annihilate(TRANSMON_LEVELS)), I(CAVITY_LEVELS))

cavity_driveR = kron(I(TRANSMON_LEVELS), create(CAVITY_LEVELS) + annihilate(CAVITY_LEVELS))
cavity_driveI = kron(I(TRANSMON_LEVELS),  1im * (create(CAVITY_LEVELS) - annihilate(CAVITY_LEVELS)))

H_drives = [transmon_driveR, transmon_driveI, cavity_driveR, cavity_driveI]

ψ1 = kron(TRANSMON_G, cavity_state(0))
ψf = kron(TRANSMON_G, cavity_state(1))

system = MultiModeQubitSystem(
    H_drift,
    H_drives, 
    ψ1,
    ψf
)

options = Options(
    max_iter = iter,
    tol = 1e-5,
    max_cpu_time = 10000.0,
)

T = 6200
Δt = 0.5
R = .5
Q = 200.0

prob = QubitProblem(
    system, 
    T;
    Δt = Δt,
    Q = Q,
    R = R,
    options = options
)

plot_multimode_qubit(system, prob.trajectory, plot_path)




