using Pico 
using SparseArrays
using HDF5

save_dir = "data/bose-hubbard"
plot_dir = "plots/bose-hubbard"

N = 2 
sites = 7 

deltas =  2π *[-0.30042389000000025,
                0.2938431699999997,
                -0.24417830999999968,
                0.25796889999999983,
                -0.1970775800000002,
                0.35186251000000013,
                -0.22094924000000038]

Js = 2π*1e-3*[9.0625, 9.032, 8.842, 8.936, 9.023, 9.040, 9.040]
U = -2π * 240e-3

H_drive = [sparse(number(N, i, sites)) for i = 1:sites]

function build_H_drift(N, sites)
    H_drift = zeros(N^sites, N^sites)
    for i = 1:sites
        H_drift += U/2 * quad(N, i, sites)
    end

    for i = 1:sites-1
        H_drift += Js[i] * create(N, i, sites)*annihilate(N, i+1, sites) +
                        create(N, i+1, sites)*annihilate(N, i, sites)
    end
    H_drift = sparse(H_drift)
end

ψ1 = zeros(N^sites)
ψ1[N+1] = 1.
ψf = zeros(N^sites)
ψf[N^(sites - 1) + 1] = 1.

control_bounds = 2π*[0.5 for i = 1:sites]


system = QuantumSystem(
    build_H_drift(N, sites),
    H_drive,
    ψ1,
    ψf, 
    a_bounds=control_bounds,
)

iter = 1
T = 100
Δt = 6.
Q = 200.
R = 0.1
cost = :infidelity_cost
eval_hess = true
pinqstate = false

options = Options(
    max_iter = iter,
    max_cpu_time = 100000.0,
)



prob = QuantumControlProblem(
    system,
    T=T,
    Δt = Δt,
    Q = Q,
    R = R,
    eval_hessian = eval_hess,
    cost = cost,
    pin_first_qstate = pinqstate,
    options = options,
    zero_a₀ = false,
    a₀s = deltas
)

experiment =
    "bose_hubbard_" *
    "T_$(prob.trajectory.T)_" *
    "dt_$(prob.trajectory.Δt)_" *
    "Q_$(prob.params[:Q])_" *
    "R_$(prob.params[:R])_" *
    "iter_$(prob.params[:options].max_iter)"

save_path = generate_file_path(
    "jld2",
    experiment,
    save_dir
)

plot_path = generate_file_path(
    "png",
    experiment,
    plot_dir
)

solve!(prob, save_path = save_path)

plot_bose_hubbard(
    system,
    prob.trajectory,
    N,
    sites,
    plot_path;
    fig_title = "Bose-Hubbard"
)

infidelity = iso_infidelity(final_state(prob.trajectory, system), ket_to_iso(ψf))
println(infidelity)
raw_controls = jth_order_controls(prob.trajectory, system, 0, d2pi = false)
save_controls = permutedims(reduce(hcat, map(Array, raw_controls)), [2,1])

result = Dict(
    "total_time" => T * Δt,
    "T" => T,
    "delta_t" => Δt,
    # "trajectory" => prob.trajectory,
    #"a_max" => amax,
    "controls" => save_controls,
    "infidelity" => infidelity
    )

save_h5_path = generate_file_path("h5", experiment, "pulses/bose_hubbard/")
println("Saving this optimization to $(save_h5_path)")
h5open(save_h5_path, "cw") do save_file
    for key in keys(result)
        write(save_file, key, result[key])
    end
end

println(prob.trajectory.times)