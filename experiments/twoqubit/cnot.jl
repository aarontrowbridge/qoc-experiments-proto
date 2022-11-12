using Pico
using LinearAlgebra

iter = 5000

experiment = "cnot_iter_$(iter)"
plot_dir = "plots/twoqubit/cnot"
plot_path = generate_file_path("png", experiment, plot_dir)

ω1 =  2π * 3.5 #GHz
ω2 = 2π * 3.9 #GHz
J = 0.1 * 2π

ψ1 = [
    [0, 0, 0, 1 + 0im],
    [1.0  + 0im, 0, 0, 0],
    [0, 1. + 0im, 0, 0],
    [0, 0, 1 + 0im, 0],
    [0.0, 1/sqrt(2) + 0im, 1/sqrt(2), 0],
    [1/sqrt(2), 0.0, 1/sqrt(2) + 0im, 0],
    [1/sqrt(2), 0.0, -1im/sqrt(2), 0.0],
    [0.0, 1/sqrt(2), 0.0, 1/sqrt(2) + 0im],
    [0.0, 1/sqrt(2), 0.0, 1im/sqrt(2) + 0im],
    [1/sqrt(2), 0.0, 0.0, -1/sqrt(2) + 0im]
]

ψf = apply.(:CX, ψ1)

# ψf = [[0, 0, 1. + 0im, 0.],
# [1.0  + 0im, 0, 0, 0],
# [0, 1. + 0im, 0, 0],
# [0, 0, 0, 1 + 0im],
# [0.0, 1/sqrt(2) + 0im, 0, 1/sqrt(2)]
# ]

    # H_drift = -(ω1/2 + gcouple)*kron(GATES[:Z], I(2)) -
    #            (ω2/2 + gcouple)*kron(I(2), GATES[:Z]) +
    #            gcouple*kron(GATES[:Z], GATES[:Z])
    # H_drift = ω1*kron(number(2), I(2)) + ω2*kron(I(2), number(2)) +
    #            J*(kron(GATES[:Z], GATES[:Z]))
    # H_drift = J*kron(GATES[:Z], GATES[:Z])

H_drift = zeros(4,4) #ω1/2 * kron(GATES[:Z], I(2)) + ω2/2 * kron(I(2), GATES[:Z]

H_drive = [
    kron(create(2), annihilate(2)) + kron(annihilate(2), create(2)),
    1im*(kron(create(2), annihilate(2)) - kron(annihilate(2), create(2))),
    kron(create(2) + annihilate(2), I(2)),
    kron(I(2), create(2) + annihilate(2))
]

#H_drive = [kron(create(2) + annihilate(2), I(2)), kron(I(2), create(2) + annihilate(2)), kron(I(2), number(2))]

control_bounds = fill(2π * 0.5, length(H_drive))
u_bounds = fill(0.5, length(H_drive))

system = QuantumSystem(
    H_drift,
    H_drive,
    ψ1,
    ψf;
    a_bounds=control_bounds,
    # u_bounds=u_bounds,
)

T = 100
Δt = 0.2
Q = 200.
R = 0.01
cost = :infidelity_cost
eval_hess = true
pinqstate = false
mode = :free_time

options = Options(
    max_iter = iter,
    tol = 1e-8,
    max_cpu_time = 7200.0
)

prob = QuantumControlProblem(
    system;
    T=T,
    mode=mode,
    equal_Δts=true,
    Δt_max=Δt,
    Δt = Δt,
    Q = Q,
    R = R,
    eval_hessian = eval_hess,
    cost = cost,
    pin_first_qstate = pinqstate,
    options = options
)

save_dir = "data/twoqubit/cnot_update"
save_path = generate_file_path("jld2", experiment, save_dir)

plot_twoqubit(
    system,
    prob.trajectory,
    plot_path;
    fig_title = "CNOT gate",
    i = 4
)

solve!(prob; save_path=save_path)

plot_twoqubit(
    system,
    prob.trajectory,
    plot_path;
    fig_title = "CNOT gate",
    i = 4
)


#infidelity = iso_infidelity(final_statei(prob.trajectory, system, i = 4), ket_to_iso(ψf[4]))
# for i in 1:4
#     display(final_state_i(prob.trajectory, system, i))
#     infidelity = iso_infidelity(final_state_i(prob.trajectory, system, i), ket_to_iso(ψf[i]))
#     println("Infidelity = $infidelity" )
# end
#display(final_statei(prob.trajectory, system, i = 3))
