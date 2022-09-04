using Pico

iter = 5000

const EXPERIMENT_NAME = "twoqubit"
plot_path = generate_file_path("png", EXPERIMENT_NAME * "_iter_$(iter)", "plots/twoqubit/")

ω1 =  2π * 1.0 #GHz
ω2 = 2π * 1.0 #GHz
J = 0.1 * 2π

ψ1 = [[1.0  + 0im, 0, 0, 0],
      [0, 1. + 0im, 0, 0],
      [0, 0, 1 + 0im, 0],
      [0, 0, 0, 1 + 0im]]
ψf = apply.(:CX, ψ1)

system = TwoQubitSystem(
    ω1 = ω1,
    ω2 = ω2,
    J = J,
    ψ1 = ψ1,
    ψf = ψf
)

T = 200
Δt = 0.1
Q = 200.
R = 0.1
loss = amplitude_loss
eval_hess = true
pinqstate = true

options = Options(
    max_iter = iter,
    tol = 1e-5,
    max_cpu_time = 7200.0
)

prob = QuantumControlProblem(
    system;
    T=T,
    Δt = Δt,
    Q = Q,
    R = R,
    eval_hessian = eval_hess,
    loss = loss,
    a_bound = 2π * 0.05,
    pin_first_qstate = pinqstate,
    options = options
)


solve!(prob)
plot_twoqubit(
    system,
    prob.trajectory,
    plot_path;
    fig_title = "sqrtiSWAP gate"
)


infidelity = iso_infidelity(final_state_i(prob.trajectory, system, i = 3), ket_to_iso(ψf[3]))
for j in 1:4
    display(final_state_i(prob.trajectory, system, i = j))
end
#display(final_statei(prob.trajectory, system, i = 3))
println("Infidelity = $infidelity" )
