using QubitControl

σx = GATES[:X]
σz = GATES[:Z]

H_drift = σz / 2
H_drive = σx / 2

gate = Symbol(ARGS[1])

ψ0 = [1, 0]
ψ1 = [0, 1]

ψ = [ψ0, ψ1, (ψ0 + im * ψ1) / √2, (ψ0 - ψ1) / √2]

# ψ = [ψ0, ψ1, im * ψ0, im * ψ1]


system = SingleQubitSystem(H_drift, H_drive, gate, ψ)

T    = 1000
Δt   = 0.01
σ    = 1.0
Q    = 0.0
Qf   = 500.0
R    = 0.001
loss = amplitude_loss
hess = false

options = Options(
    max_iter = 200,
    tol = 1e-6
)

iter = parse(Int, ARGS[2])

tol = parse(Float64, ARGS[3])

Rᵤ = 1.0e-6
Rₛ = Rᵤ

plot_dir = "plots/single_qubit/min_time"

plot_file = "$(gate)_gate_iter_$(iter)_tol_$(tol)_Ru_$(Rᵤ)_Rs_$(Rₛ)_pinned.png"

plot_path = joinpath(plot_dir, plot_file)

min_time_options = Options(
    max_iter = iter,
    tol = tol
)

prob = MinTimeProblem(
    system,
    T;
    Δt=Δt,
    σ=σ,
    Q=Q,
    Qf=Qf,
    R=R,
    Rᵤ=Rᵤ,
    Rₛ=Rₛ,
    eval_hessian=hess,
    loss=loss,
    options=options,
    min_time_options=min_time_options
)

plot_single_qubit_2_qstate_with_seperated_controls(
    prob.subprob.trajectory,
    plot_path,
    system.isodim,
    system.control_order,
    T;
    fig_title="min time $gate gate on basis states (iter = $iter; tol = $tol)"
)

solve!(prob)

plot_single_qubit_2_qstate_with_seperated_controls(
    prob.subprob.trajectory,
    plot_path,
    system.isodim,
    system.control_order,
    T;
    fig_title="min time $gate gate on basis states (iter = $iter; tol = $tol)"
)
