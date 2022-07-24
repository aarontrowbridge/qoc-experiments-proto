using QubitControl 

iter = 2000

const EXPERIMENT_NAME = "transmon"
plot_path = generate_file_path("png", EXPERIMENT_NAME * "_iter_$(iter)", "plots/transmon/")

#system parameters

qubit_frequency = 2π * 4.96 #GHz
anharmonicity = -2π * 0.143 #GHz

levels = 3

ψg = [1. + 0*im, 0 , 0]
ψe = [0, 1. + 0*im, 0]

ψ1 = [ψg, ψe]
ψf = [-im*ψe, -im*ψg]

system = TransmonSystem(
    levels = levels, 
    rotating_frame = true,
    ω = qubit_frequency,
    α = anharmonicity,
    ψ1 = ψ1,
    ψf = ψf
)

#T is number of time steps, not total time
T = 400
Δt = 0.1 
Q = 0.0
Qf = 200.0
R = 0.1
loss = amplitude_loss
hess = false

options = Options(
    max_iter = iter,
    tol = 1e-5
)

prob = QubitProblem(
    system,
    T;
    Δt = Δt,
    Q = Q,
    Qf = Qf,
    R = R,
    eval_hessian = hess,
    loss = loss,
    a_bound = 2π * 19e-3,
    options = options
)

solve!(prob)

plot_transmon(
    system,
    prob.trajectory,
    plot_path;
    fig_title="X gate on basis states"
)