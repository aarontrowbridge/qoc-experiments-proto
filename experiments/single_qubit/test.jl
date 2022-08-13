using QubitControl
using JLD2

σx = GATES[:X]
σy = GATES[:Y]
σz = GATES[:Z]

H_drift = σz / 2
H_drive = [σx / 2, σy / 2]

gate = :X
iter = 100


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
cost = infidelity_cost
hess = true

a_bounds = [1.0, 0.5]

pin_first_qstate = false

integrator = :FourthOrderPade

experiment = "$(gate)_gate_2_controls_test_R_$(R)_T_$(T)_iter_$(iter)"

plot_dir = "plots/single_qubit/test"

save_dir = "data/single_qubit/test/problems"

save_path = generate_file_path("jld2", experiment, save_dir)

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
    cost=cost,
    pin_first_qstate = true,
    options=options,
    integrator=integrator,
)

# solve for first n iters

solve!(prob; save=true, path=save_path)

plot_file = experiment * "_pre_save"

plot_path = generate_file_path("png", plot_file, plot_dir)

plot_single_qubit(
    system,
    prob.trajectory,
    plot_path,
    fig_title="$gate gate on basis states"
)

loaded_prob = load_object(save_path)

solve!(loaded_prob; save=false)

plot_file = experiment * "_post_save"

plot_path = generate_file_path("png", plot_file, plot_dir)


plot_single_qubit(
    system,
    prob.trajectory,
    plot_path,
    fig_title="$gate gate on basis states"
)

infidelity = iso_infidelity(final_state2(prob.trajectory, system), ket_to_iso(apply(:X, ψ1)))
println(infidelity)
