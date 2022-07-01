using QubitControl

σx = GATES[:X]
σz = GATES[:Z]

H_drift = σz / 2
H_drive = σx / 2

gate = :X

ψ0 = [1, 0]
ψ1 = [0, 1]

ψ = [ψ0, ψ1]

system = SingleQubitSystem(
    H_drift,
    H_drive,
    gate, ψ
)

T    = 1000
Δt   = 0.01
σ    = 1.0
Q    = 0.0
Qf   = 500.0
R    = 0.001
loss = quaternionic_loss
hess = false

prob = QubitProblem(
    system,
    T;
    Δt=Δt,
    σ=σ,
    Q=Q,
    Qf=Qf,
    R=R,
    eval_hessian=hess,
    loss=loss
)

solve!(prob)

plot_path = "plots/single_qubit/test/$(gate)_gate.png"

plot_single_qubit_2_qstate_with_controls(
    prob,
    plot_path;
    fig_title="$gate gate on basis states"
)
