using Pico
using NamedTrajectories
using JLD2

# data_path = "data/multimode/good_solutions/g0_to_g1_T_101_dt_4.0_Q_500.0_R_0.1_u_bound_1.0e-5.jld2"
data_path = "data/multimode/free_time/no_guess/problems/g0_to_g1_T_100_dt_10.0_Δt_max_factor_1.0_Q_1000.0_R_1.0e-5_iter_2000_u_bound_1.0e-6_alpha_transmon_20.0_alpha_cavity_20.0_resolve_2_00000.jld2"

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

bounds = (;
    a = data.system.a_bounds,
    # dda = data.params[:u_bounds]
)

Ẑ = Traj(
    components;
    controls=:dda,
    # controls=:a,
    bounds=bounds,
    dt=Δts[end],
    initial=(a=zeros(4),),
    final=(a=zeros(4),)
)

# save the trajectory

@save "test_data.jld2" Ẑ




P = FourthOrderPade(data.system)

function dynamics(
    zₜ::AbstractVector{<:Real},
    zₜ₊₁::AbstractVector{<:Real},
    Z::Traj
)
    xₜ = zₜ[Z.components.states]
    xₜ₊₁ = zₜ₊₁[Z.components.states]
    uₜ = zₜ[Z.components.controls]
    Δt = Z.dt
    return Pico.Dynamics.dynamics(xₜ₊₁, xₜ, uₜ, Δt, P, data.system)
end

function dynamics2(
    zₜ::AbstractVector{<:Real},
    zₜ₊₁::AbstractVector{<:Real},
    Z::Traj
)
    ψ̃ₜ = zₜ[Z.components.ψ̃]
    ψ̃ₜ₊₁ = zₜ₊₁[Z.components.ψ̃]
    aₜ = zₜ[Z.components.a]
    Δt = Z.dt
    return P(ψ̃ₜ₊₁, ψ̃ₜ, aₜ, Δt)
end

transmon_levels = 3
cavity_levels = 14
ψ1 = "g0"
ψf = "g1"
χ = 1.0 * data.system.params[:χ]

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

a_bounds = data.system.a_bounds

function quantize(A::Matrix{Float64})
    res = a_bounds ./ 128
    Â = similar(A)
    for i = 1:size(A, 2)
        Â[:, i] = round.(A[:, i] ./ res) .* res
    end
    return Â
end

# τs = 1:Ẑ.T
τs = [Ẑ.T]

experiment = QuantumExperiment(
    experimental_system,
    Ẑ[1].ψ̃,
    x -> x,
    # g_pop,
    τs;
    integrator=exp,
    control_transform=quantize,
)

max_iter = 20
max_backtrack_iter = 15
fps = 2
α = 0.5
β = 1.0
R = (
    a   = 1.0e1,
    dda = 1.0e10
)
Qy = 1.0e-5
Qyf = 1.0e3

QP_tol = 1e-6
QP_max_iter = 1e5
QP_settings = Dict(
    :polish => true,
)

static = false
nominal_correction = false
dynamic_measurements = false
correction_term = false

prob = ILCProblemNew(
    Ẑ,
    dynamics,
    experiment;
    max_iter=max_iter,
    QP_verbose=false,
    correction_term=correction_term,
    dynamic_measurements=dynamic_measurements,
    norm_p=1,
    R=R,
    static_QP=static,
    Qy=Qy,
    Qyf=Qyf,
    α=α,
    β=β,
    max_backtrack_iter=max_backtrack_iter,
    QP_tol=QP_tol,
    QP_max_iter=QP_max_iter,
    QP_settings=QP_settings,
    mle=true,
)

solve!(prob; nominal_correction=nominal_correction)

plot_dir = "plots/multimode/ILC/refactor"

plot_name =
    (static ? "static_" : "dynamic_") *
    "T_$(Ẑ.T)_taus_$(length(τs))_" *
    "alpha_$(α)_beta_$(β)_" *
    "R_" * join([string(k) * "_" * string(v) for (k, v) in pairs(R)], "_") * "_" *
    "Qy_$(Qy)_Qyf_$(Qyf)_QP_tol_$(QP_tol)" *
    (nominal_correction ? "_nominal_correction" : "")

plot_path = generate_file_path("gif", plot_name, plot_dir)

animate_ILC_multimode(prob, plot_path; fps=fps)
