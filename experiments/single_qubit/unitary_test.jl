using Pico
using JLD2

H_drift = zeros(2,2)
H_drive = [create(2) + annihilate(2), 1im*(annihilate(2) - create(2))]

iter = 500 

U_goal = [1/sqrt(2) -im/sqrt(2); 
          -im/sqrt(2) 1/sqrt(2)]
 

control_bounds = [2π * 19e-3,  2π * 19e-3]

system = QuantumSystem(
    H_drift,
    H_drive,
    U_goal,
    control_bounds
)

T = 400
Δt = 0.1
Q = 200.
R = 0.1
cost = :frobenius_cost
hess = true
pinqstate = false

options = Options(
    max_iter = iter,
    tol = 1e-5
)

prob = QuantumControlProblem(
    system;
    T=T,
    Δt = Δt,
    Q = Q,
    R = R,
    eval_hessian = hess,
    cost = cost,
    pin_first_qstate = pinqstate,
    options = options
)

experiment = "Rx(piby2)_gate_2_controls_test_R_$(R)_T_$(T)_iter_$(iter)"

plot_dir = "plots/single_qubit/unitary_test/"

plot_path = generate_file_path("png", experiment, plot_dir)

plot_single_qubit(
    prob.system,
    prob.trajectory,
    plot_path,
    fig_title="Rx(piby2) gate gate on basis states"
)

solve!(prob)

display(final_unitary(prob.trajectory, system))

plot_single_qubit(
    prob.system,
    prob.trajectory,
    plot_path,
    fig_title="Rx(piby2) gate gate on basis states"
)