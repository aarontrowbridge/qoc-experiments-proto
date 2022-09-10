using Pico

σx = GATES[:X]
σy = GATES[:Y]
σz = GATES[:Z]

H_drift = σz / 2
H_drives = [σx / 2, σy / 2]

gate = :X

ψ0 = [1, 0]
ψ1 = [0, 1]

ψ = [ψ0, ψ1, (ψ0 + im * ψ1) / √2, (ψ0 - ψ1) / √2]

ψf = apply.(gate, ψ)

a_bounds = [1.0, 0.5]

system = QuantumSystem(
    H_drift,
    H_drives,
    ψ,
    ψf,
    a_bounds
)

T  = 1000
Δt = 0.01
Q  = 200.0
R  = 0.01
Rᵤ = 1e-10
Rₛ = Rᵤ

pin_first_qstate = true

iter = 30

options = Options(
    max_iter = iter,
)

plot_dir = "plots/single_qubit/mintime"
experiment = "$(gate)_gate_Ru_$(Rᵤ)_Rs_$(Rₛ)_iter_$(iter)"
plot_path = generate_file_path("png", experiment, plot_dir)

mintime_iter = 100

mintime_options = Options(
    max_iter = mintime_iter,
)

prob = MinTimeQuantumControlProblem(
    system;
    T=T,
    Δt=Δt,
    Q=Q,
    R=R,
    Rᵤ=Rᵤ,
    Rₛ=Rₛ,
    pin_first_qstate=pin_first_qstate,
    options=options,
    mintime_options=mintime_options
)

plot_single_qubit(
    system,
    prob.subprob.trajectory,
    plot_path
)

save_dir = "data/single_qubit/mintime/problems"
save_path = generate_file_path("jld2", experiment, save_dir)

solve!(prob; save_path=save_path)

plot_single_qubit(
    system,
    prob.subprob.trajectory,
    plot_path
)

loaded_prob = load_problem(save_path)

solve!(loaded_prob)

plot_single_qubit(
    loaded_prob.subprob.system,
    loaded_prob.subprob.trajectory,
    generate_file_path("png", experiment * "_resolved", plot_dir)
)
