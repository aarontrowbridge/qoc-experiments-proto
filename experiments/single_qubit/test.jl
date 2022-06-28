using QubitControl

σx = GATES[:X]
σz = GATES[:Z]

H_drift = σz / 2
H_drive = σx / 2

gate = :X

ψ0 = [1, 0]

system = SingleQubitSystem(H_drift, H_drive, gate, ψ0)

T    = 1000
Δt   = 0.01
σ    = 1.0
Q    = 0.1
Qf   = 2500.0
R    = 0.00000001
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
    loss=quaternionic_loss
)

plot_path = "plots/single_qubit/pretest.png"
plot_single_qubit_with_controls(prob, plot_path; fig_title="pretest")

solve!(prob)

plot_path = "plots/single_qubit/test.png"
plot_single_qubit_with_controls(prob, plot_path; fig_title="test")
