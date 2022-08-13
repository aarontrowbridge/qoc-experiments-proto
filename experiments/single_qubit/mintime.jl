using QubitControl

σx = GATES[:X]
σy = GATES[:Y]
σz = GATES[:Z]

H_drift = σz / 2
H_drives = [σx / 2, σy / 2]

gate = Symbol(:X)

ψ0 = [1, 0]
ψ1 = [0, 1]

ψ = [ψ0, ψ1, (ψ0 + im * ψ1) / √2, (ψ0 - ψ1) / √2]
ψf = apply.(gate, ψ)
a_bound = [1.0, 0.5]

system = QuantumSystem(
    H_drift, 
    H_drives, 
    ψ1 = ψ,
    ψf = ψf,
    control_bounds = a_bound
)

T  = 1000
Δt = 0.01
Q  = 200.0
R  = 0.01
Rᵤ = 1e-5
Rₛ = Rᵤ
cost = infidelity_cost


iter = 3500
tol = 1e-5

options = Options(
    max_iter = iter,
    tol = tol,
    max_cpu_time = 10000.0,
)



plot_dir = "plots/single_qubit/min_time/two_controls"
plot_file = "$(gate)_gate_Ru_$(Rᵤ)_Rs_$(Rₛ)_tol_$(tol)_iter_$(iter)_pinned.png"
plot_path = joinpath(plot_dir, plot_file)

min_time_options = Options(
    max_iter = 14000,
    tol = tol,
    max_cpu_time = 27000.0,
)

prob = QuantumMinTimeProblem(
    system,
    T;
    Δt=Δt,
    Q=Q,
    R=R,
    Rᵤ=Rᵤ,
    Rₛ=Rₛ,
    cost=cost,
    options=options,
    mintime_options=min_time_options
)

plot_single_qubit(
    system,
    prob.subprob.trajectory,
    plot_path
)

solve!(prob)

plot_single_qubit(
    system,
    prob.subprob.trajectory,
    plot_path
)

