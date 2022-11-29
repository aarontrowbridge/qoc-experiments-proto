# using Pkg
# ENV["PYTHON"] = Sys.which("python") 
# Pkg.build("PyCall")
using Pico
using PyCall
using Statistics
using JLD2

@pyinclude "experiments/multimode/run_experiment_optimize_loop.py"

println("howdy")


data_dir = "data_tracked/multimode/good_solutions"

data_name = "g0_to_g1_T_101_dt_4.0_Q_500.0_R_0.1_u_bound_1.0e-5"

data_path = joinpath(data_dir, data_name * ".jld2")

data = load_data(data_path)

xs = [
    data.trajectory.states[t][1:data.system.isodim]
        for t = 1:data.trajectory.T
]

us = [
    data.trajectory.states[t][
        (data.system.n_wfn_states +
        data.system.∫a * data.system.ncontrols) .+
        (1:data.system.ncontrols)
    ] for t = 1:data.trajectory.T
]

Ẑ = Trajectory(
    xs,
    us,
    data.trajectory.times,
    data.trajectory.T,
    data.trajectory.Δt
)

T = data.trajectory.T

cavity_levels = 14

function g_pop(x)
    y = []
    append!(y, sum(x[1:cavity_levels].^2 + x[3*cavity_levels .+ (1:cavity_levels)].^2))
    append!(
        y,
        sum(
            x[cavity_levels .+ (1:cavity_levels)].^2 +
            x[4*cavity_levels .+ (1:cavity_levels)].^2
        )
    )
    for i = 1:10
        append!(y,
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

ydim = length(g_pop(data.trajectory.states[1]))

τs = [T]

function g_hardware(
    us::Vector{Vector{Float64}},
    times::AbstractVector{Float64},
    τs::AbstractVector{Int}
)::MeasurementData
    # display(us) 
    intermediate = length(τs) == 1 ? false : true
    ys = py"take_controls_and_measure"(times, us, τs, intermediate) |> transpose
    display(ys)
    println(typeof(ys))
    ys = collect(eachcol(ys)) 
    return MeasurementData(ys, τs, ydim)
end
    




# function build_cov_matrix(N=20)
#     us = Ẑ.actions
#     ys = []
#     for _ = 1:N
#         y = experiment(us).ys[end]
#         push!(ys, y)
#     end
#     Y = hcat(ys...)
#     return cov(Y; dims=2)
# end

# Σ = build_cov_matrix(15)

# @info "cov matrix"
# display(Σ)

max_iter = 10 
max_backtrack_iter = 10
fps = 2
α = 0.5
β = 0.01
R = 1.0e0
Qy = 1.0e1
Qf = 1.0e2

for τs in [ 
    # [Ẑ.T],
    [25, Ẑ.T], 
    [25, 50, Ẑ.T], 
    [25, 50, 75, Ẑ.T], 
] 
    save_dir = "hardware_ILC"
    save_name = "taus_" * join(τs, "_")

    experiment = HardwareExperiment(
        g_hardware,
        g_pop,
        τs,
        data.trajectory.times,
        ydim
    )

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
        # Σ=Σ,
        max_backtrack_iter=max_backtrack_iter,
    )



    log_dir = "log/multimode/hardware/ILC"
    log_path = generate_file_path("jld2", save_name, log_dir)

    solve!(prob)

    @save log_path prob

    plot_dir = "plots/multimode/hardware/ILC"
    plot_path = generate_file_path("gif", save_name, plot_dir)

    animate_ILC_multimode(prob, plot_path; fps=fps)
end