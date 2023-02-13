using PicoOld
using LinearAlgebra

iter = 5000

const EXPERIMENT_NAME = "sqrtiSWAP"
plot_path = generate_file_path("png", EXPERIMENT_NAME * "_iter_$(iter)", "plots/twoqubit/")

ω1 =  2π * 3.5 #GHz
ω2 = 2π * 3.9 #GHz
J = 0.1 * 2π

ψ1 = [[0, 0, 0, 1 + 0im],
      [1.0  + 0im, 0, 0, 0],
      [0, 1. + 0im, 0, 0],
      [0, 0, 1 + 0im, 0]
      ]
ψf = apply.(:sqrtiSWAP, ψ1)


    # H_drift = -(ω1/2 + gcouple)*kron(GATES[:Z], I(2)) -
    #            (ω2/2 + gcouple)*kron(I(2), GATES[:Z]) +
    #            gcouple*kron(GATES[:Z], GATES[:Z])
    # H_drift = ω1*kron(number(2), I(2)) + ω2*kron(I(2), number(2)) +
    #            J*(kron(GATES[:Z], GATES[:Z]))
    # H_drift = J*kron(GATES[:Z], GATES[:Z])
H_drift = zeros(4,4) #ω1/2 * kron(GATES[:Z], I(2)) + ω2/2 * kron(I(2), GATES[:Z]

H_drive = [kron(create(2), annihilate(2)) + kron(annihilate(2), create(2)),
           1im*(kron(create(2), annihilate(2)) - kron(annihilate(2), create(2)))]

           #H_drive = [kron(create(2) + annihilate(2), I(2)), kron(I(2), create(2) + annihilate(2)), kron(I(2), number(2))]

control_bounds = [2π * 0.5 for x in 1:length(H_drive)]

system = QuantumSystem(
    H_drift,
    H_drive,
    ψ1,
    ψf,
    control_bounds
)

T = 100
Δt = 0.1
Q = 200.
R = 0.1
cost = :infidelity_cost
eval_hess = true
pinqstate = true

options = Options(
    max_iter = iter,
    tol = 1e-8,
    max_cpu_time = 7200.0
)

prob = QuantumControlProblem(
    system,
    T;
    Δt = Δt,
    Q = Q,
    R = R,
    eval_hessian = eval_hess,
    cost = cost,
    pin_first_qstate = pinqstate,
    options = options
)


solve!(prob)
plot_twoqubit(
    system,
    prob.trajectory,
    plot_path;
    fig_title = "sqrtiSWAP gate",
    i = 4
)


infidelity = iso_infidelity(final_statei(prob.trajectory, system, i = 4), ket_to_iso(ψf[4]))
for j in 1:4
    display(final_statei(prob.trajectory, system, i = j))
end
#display(final_statei(prob.trajectory, system, i = 3))
println("Infidelity = $infidelity" )
