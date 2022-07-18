using QubitControl

σx = GATES[:X]
σy = GATES[:Y]
σz = GATES[:Z]

H_drift = σz / 2
H_drive = [σx / 2, σy / 2]

gate = Symbol(ARGS[1])

plot_path = "plots/single_qubit/test/$(gate)_gate_2_controls_test.png"

ψ0 = [1, 0]
ψ1 = [0, 1]


# ψ = [ψ0, ψ1, (ψ0 + ψ1) / √2]

# ψ = (ψ0 + ψ1) / √2

ψ = [ψ0, ψ1, (ψ0 + im * ψ1) / √2, (ψ0 - ψ1) / √2]

system = SingleQubitSystem(
    H_drift,
    H_drive,
    gate, ψ;
    control_order=2
)

T    = 1000
Δt   = 0.01
Q    = 0.0
Qf   = 200.0
R    = 0.1
loss = amplitude_loss
hess = false
iter = parse(Int, ARGS[2])

options = Options(
    max_iter = iter,
    tol = 1e-8
)

prob = QubitProblem(
    system,
    T;
    Δt=Δt,
    Q=Q,
    Qf=Qf,
    R=R,
    eval_hessian=hess,
    loss=loss,
    options=options
)

plot_single_qubit(
    system,
    prob.trajectory,
    plot_path;
    fig_title="$gate gate on basis states"
)

solve!(prob)

plot_single_qubit(
    system,
    prob.trajectory,
    plot_path,
    fig_title="$gate gate on basis states"
)
