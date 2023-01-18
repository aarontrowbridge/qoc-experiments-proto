using Pico

# data_dir = "data/multimode/fixed_time_update/guess/pinned/problems"
data_dir = "experiments/multimode"

# data_name = "g0_to_g1_T_102_dt_4.0_Q_500.0_R_0.1_iter_2000_u_bound_1.0e-5_alpha_transmon_20.0_alpha_cavity_20.0_resolve_5_00000"

data_name = "g0_to_g1_T_101_dt_4.0_Q_500.0_R_0.1_u_bound_1.0e-5"

data_path = joinpath(data_dir, data_name * ".jld2")

data = load_data(data_path)

xs = [
    data.trajectory.states[t]
        for t = 1:data.trajectory.T
]

us = [
    data.trajectory.actions[t]
        for t = 1:data.trajectory.T
]

udim = length(us[1])

for i = 1:data.trajectory.T-1
    us[i] = us[i+1]
end

Ẑ = Trajectory(
    xs,
    us,
    data.trajectory.times,
    data.trajectory.T,
    data.trajectory.Δt
)

transmon_levels = 3
cavity_levels = 14
ψ1 = "g0"
ψf = "g1"
χ = 1.2 * data.system.params[:χ]

experimental_system = MultiModeSystem(
    transmon_levels,
    cavity_levels,
    ψ1,
    ψf;
    χ=χ
)

g(ψ̃) = abs2.(iso_to_ket(ψ̃))

function g_pop(x)
    y = []
    append!(y,
        sum(
            x[1:cavity_levels].^2 +
            x[3*cavity_levels .+ (1:cavity_levels)].^2
        )
    )
    append!(y,
        sum(
            x[cavity_levels .+ (1:cavity_levels)].^2 +
            x[4*cavity_levels .+ (1:cavity_levels)].^2
        )
    )
    for i = 1:10
        append!(
            y,
            x[i]^2 +
            x[i + 3 * cavity_levels]^2 +
            x[i + cavity_levels]^2 +
            x[i + 4 * cavity_levels]^2 +
            x[i + 2 * cavity_levels]^2 +
            x[i + 5 * cavity_levels]^2
        )
        #append!(y, x[i + cavity_levels]^2 + x[i+3*cavity_levels]^2)
    end
    return convert(typeof(x), y)
end

experiment = QuantumExperiment(
    experimental_system,
    Ẑ.states[1],
    Ẑ.times,
    # x -> x,
    g_pop,
    # [5:5:50; Ẑ.T];
    #[10:10:100; Ẑ.T];
    [25, 50, 75, Ẑ.T];
    # [50, Ẑ.T];
    # [2:2:Ẑ.T - 10; Ẑ.T];
    # [1:Ẑ.T ÷ 2; Ẑ.T];
    # 1:Ẑ.T;
    integrator=exp,
    d2u=true,
)

max_iter = 20
max_backtrack_iter = 15
fps = 2
α = 0.5
β = 1.0
R = 1.0e2
Qy = 1.0e2
Qf = 1.0e2


prob = ILCProblem(
    data.system,
    Ẑ,
    experiment;
    max_iter=max_iter,
    QP_verbose=false,
    correction_term=true,
    norm_p=1,
    R=R,
    static_QP=false,
    Qy=Qy,
    Qf=Qf,
    use_system_goal=true,
    α=α,
    β=β,
    max_backtrack_iter=max_backtrack_iter,
    d2u=true,
    d2u_bounds=fill(1e-6, 4),
    QP_tol = 1e-3
)

solve!(prob)

plot_dir = "plots/multimode/good_solutions/ILC"
plot_path = generate_file_path("gif", data_name, plot_dir)

animate_ILC_multimode(prob, plot_path; fps=fps)
