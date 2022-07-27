using QubitControl

σx = GATES[:X]
σy = GATES[:Y]
σz = GATES[:Z]

H_drift = σz / 2
H_drive = [σx / 2, σy / 2]

gate = Symbol(ARGS[1])
iter = parse(Int, ARGS[end])


ψ0 = [1, 0]
ψ1 = [0, 1]


# ψ = [ψ0, ψ1, (ψ0 + ψ1) / √2]

# ψ = (ψ0 + ψ1) / √2

ψ = [ψ0, ψ1, (ψ0 + im * ψ1) / √2, (ψ0 - ψ1) / √2]

system = SingleQubitSystem(
    H_drift,
    H_drive,
    gate, ψ
)

T    = parse(Int, ARGS[2])
Δt   = 0.01
Q    = 200.0
R    = 2.0
loss = amplitude_loss
hess = true

plot_dir = "plots/single_qubit/test"
plot_file = "$(gate)_gate_2_controls_test_R_$(R)_T_$(T)_iter_$(iter).png"

plot_path = joinpath(plot_dir, plot_file)

options = Options(
    max_iter = iter,
    tol = 1e-5,
    # linear_solver="pardiso",
)

prob = QubitProblem(
    system,
    T;
    Δt=Δt,
    Q=Q,
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
