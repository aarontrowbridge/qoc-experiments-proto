using Pico

data_path = "data/multimode/good_solutions/g0_to_g1_T_101_dt_4.0_Q_500.0_R_0.1_u_bound_1.0e-5.jld2"

data = load_data(data_path)

n_wfn_states = data.system.n_wfn_states
ncontrols = data.system.ncontrols

Ψ̃ = hcat([x[1:n_wfn_states] for x in data.trajectory.states]...)
A = hcat([x[n_wfn_states .+ slice(1, ncontrols)] for x in data.trajectory.states]...)
dA = hcat([x[n_wfn_states .+ slice(2, ncontrols)] for x in data.trajectory.states]...)
ddA = hcat(data.trajectory.actions...)
ts = data.trajectory.times
Δts = [[ts[t] - ts[t-1] for t = 2:data.trajectory.T]; data.trajectory.Δt]

components = (
    ψ̃ = Ψ̃,
    a = A,
    da = dA,
    dda = ddA,
    # Δt = Δts
)

bounds = (
    a = data.system.a_bounds,
    dda = data.params[:u_bounds]
)

Ẑ = Traj(
    components;
    controls=:dda,
    bounds=bounds,
    dt=Δts[end],
    initial=(a=zeros(4), da=zeros(4)),
    final=(a=zeros(4), da=zeros(4))
)


function dynamics(
    zₜ::AbstractVector{<:Real},
    zₜ₊₁::AbstractVector{<:Real},
    Z::Traj
)
    xₜ = zₜ[Z.components.states]
    xₜ₊₁ = zₜ₊₁[Z.components.states]
    uₜ = zₜ[Z.components.controls]
    Δt = Z.dt
    P = FourthOrderPade(data.system)
    return Pico.Dynamics.dynamics(xₜ, xₜ₊₁, uₜ, Δt, P, data.system)
end

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

function g_pop(ψ̃)
    y = []
    append!(y, sum(ψ̃[1:cavity_levels].^2 + ψ̃[3*cavity_levels .+ (1:cavity_levels)].^2))
    append!(
        y,
        sum(
            ψ̃[cavity_levels .+ (1:cavity_levels)].^2 +
            ψ̃[4*cavity_levels .+ (1:cavity_levels)].^2
        )
    )
    for i = 1:10
        append!(y,
            ψ̃[i]^2 +
            ψ̃[i + 3 * cavity_levels]^2 +
            ψ̃[i + cavity_levels]^2 +
            ψ̃[i + 4 * cavity_levels]^2 +
            ψ̃[i + 2 * cavity_levels]^2 +
            ψ̃[i + 5 * cavity_levels]^2
        )
        #append!(y, ψ̃[i + cavity_levels]^2 + ψ̃[i+3*cavity_levels]^2)
    end
    return convert(typeof(ψ̃), y)
end

experiment = QuantumExperiment(
    experimental_system,
    Ẑ.ψ̃[:, 1],
    # x -> x,
    g_pop,
    # [5:5:50; 75; Ẑ.T];
    # [10:10:100; Ẑ.T];
    # [25, 50, 75, Ẑ.T];
    # [50, Ẑ.T];
    # [10, 25, 50, 75, Ẑ.T];
    # [Ẑ.T];
    # [2:2:Ẑ.T - 10; Ẑ.T];
    # [1:Ẑ.T ÷ 2; Ẑ.T];
    1:Ẑ.T;
    integrator=exp
)

max_iter = 20
max_backtrack_iter = 10
fps = 2
α = 0.5
β = 0.01
R = (a=1.0e-3, dda=1.0e-3)
Qy = 1.0e1
Qyf = 2.0e2
QP_tol = 1e-6

prob = ILCProblemNew(
    Ẑ,
    dynamics,
    experiment;
    max_iter=max_iter,
    QP_verbose=false,
    correction_term=true,
    norm_p=1,
    R=R,
    static_QP=true,
    Qy=Qy,
    Qyf=Qyf,
    α=α,
    β=β,
    max_backtrack_iter=max_backtrack_iter,
    QP_tol=QP_tol,
    QP_max_iter=1e5,
    mle=true,
)

solve!(prob)

plot_dir = "plots/multimode/good_solutions/ILC"
plot_path = generate_file_path("gif", data_name, plot_dir)

animate_ILC_multimode(prob, plot_path; fps=fps)
