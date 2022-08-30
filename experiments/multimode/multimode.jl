using Pico
using LinearAlgebra
using JLD2

iter = 20000

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

cavity_a_bounds = fill(0.03, 2)

a_bounds = [qubit_a_bounds; cavity_a_bounds]

T = 1350
Δt = 2.
R = 0.1
pin_first_qstate = false
phase = 0.

system = QuantumSystem(
    H_drift,
    H_drives,
    ψ1 = ψ1,
    ψf = ψf,
    control_bounds = a_bounds,
    phase = phase
)

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

cons = AbstractConstraint[u_bounds]

experiment = "g0_to_g1_T_$(T)_dt_$(Δt)_R_$(R)_iter_$(iter)" * (pin_first_qstate ? "_pinned" : "") * "phase_$(phase)"

plot_dir = "plots/multimode/fermium"
data_dir = "data/multimode/fixed_time/no_guess/problems"



prob = QuantumControlProblem(
    system, 
    T;
    Δt=Δt,
    R=R,
    pin_first_qstate=pin_first_qstate,
    options=options,
    cons=cons
)


let sol = true, i = 0

    while sol
        resolve = "_resolve_$(i)"
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
        plot_multimode(system, prob.trajectory, plot_path)
        solve!(prob, save=true, path=data_path)
        plot_multimode(system, prob.trajectory, plot_path)

        prompt = true
        while prompt 
            println("Resolve? (y/n)")
            answer = readline()
            if answer == "y"
                global prob = load_object(data_path)
                prompt = false
            elseif answer == "n"
                prompt = false
                sol = false
            else 
                println("Invalid response, must be y or n")
            end
        end
        i +=1
    end
end
