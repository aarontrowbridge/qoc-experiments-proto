using QubitControl

σx = GATES[:X]
σz = GATES[:Z]

H_drift = σz / 2
H_drive = σx / 2

gate = :X

ψ0 = [1, 0]
ψ1 = [0, 1]

ψ = [ψ0, ψ1]

system = SingleQubitSystem(H_drift, H_drive, gate, ψ)

T    = 1000
Δt   = 0.01
σ    = 1.0
Q    = 0.0
Qf   = 500.0
R    = 0.001
loss = quaternionic_loss
hess = false

options = Options(
    max_iter = 500
)

B = 1.0
squared_loss = false
iter = parse(Int, ARGS[1])

tol = 1e-10

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
    B=B,
    eval_hessian=hess,
    loss=loss,
    squared_loss=squared_loss,
    options=options,
    min_time_options=min_time_options
)

plot_single_qubit_2_qstate_with_seperated_controls(
    prob.subprob.trajectory,
    "plots/single_qubit/min_time/$(gate)_gate_B_$(B)_iter_$(iter)_tol_$(tol)" *
        (squared_loss ? "_squared_loss" : "") * ".png",
    system.isodim,
    system.control_order,
    T;
    fig_title="min time $gate gate on basis states (iter = $iter)" *
        (squared_loss ? " w/ squared loss" : "")
)

solve!(prob)

plot_single_qubit_2_qstate_with_seperated_controls(
    prob.subprob.trajectory,
    "plots/single_qubit/min_time/$(gate)_gate_B_$(B)_iter_$(iter)_tol_$(tol)" *
        (squared_loss ? "_squared_loss" : "") * ".png",
    system.isodim,
    system.control_order,
    T;
    fig_title="min time $gate gate on basis states (iter = $iter; tol = $tol)" *
        (squared_loss ? " w/ squared loss" : "")
)
