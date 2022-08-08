using QubitControl

σx = GATES[:X]
σy = GATES[:Y]
σz = GATES[:Z]

H_drift = σz / 2
H_drive = [σx / 2, σy / 2]

gate = :X
iter = 1000


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

T    = 1000
Δt   = 0.01
Q    = 200.0
R    = 2.0
loss = infidelity_loss
hess = true

a_bounds = [1.0, 0.5]

pin_first_qstate = false

integrator = :FourthOrderPade

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
    a_bound=a_bounds,
    loss=loss,
    pin_first_qstate = true,
    options=options,
    integrator=integrator,
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

infidelity = iso_infidelity(final_state2(prob.trajectory, system), ket_to_iso(apply(:X, ψ1)))
println(infidelity)