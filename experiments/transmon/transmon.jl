using QubitControl
using HDF5

iter = 3000

const EXPERIMENT_NAME = "8-5-2022-transmon_no_int_a"
plot_path = generate_file_path("png", EXPERIMENT_NAME * "_iter_$(iter)", "plots/transmon/")

#system parameters

qubit_frequency = 2π * 4.96 #GHz
anharmonicity = -2π * 0.143 #GHz
amax = 2π*19e-3

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
Q = 200.
R = 0.1
cost = amplitude_cost
hess = true
pinqstate = true

time = T * Δt

options = Options(
    max_iter = iter,
    tol = 1e-5
)

prob = QubitProblem(
    system,
    T;
    Δt = Δt,
    Q = Q,
    R = R,
    eval_hessian = hess,
    cost = cost,
    a_bound = amax,
    pin_first_qstate = pinqstate,
    options = options
)

display(prob.trajectory.states[2])

solve!(prob)

display(prob.trajectory.states[2])

raw_controls = jth_order_controls(prob.trajectory, system, 0, d2pi = false)
display(raw_controls)
controls = permutedims(reduce(hcat, map(Array, raw_controls)), [2,1])
display(controls)

infidelity = iso_infidelity(final_state2(prob.trajectory, system), ket_to_iso(-im*ψg))
println(infidelity)

display(final_state2(prob.trajectory, system))

result = Dict(
    "Q" => Q,
    "R" => R,
    "total_time" => T * Δt,
    "T" => T,
    "delta_t" => Δt,
    "eval_hessian" => hess,
    "a_bound" => 19e-3,
    # "trajectory" => prob.trajectory,
    "a_max" => amax,
    "pin_first_qstate" => pinqstate,
    "controls" => controls,
    "infidelity" => infidelity
)


save_file_path = generate_file_path("h5", EXPERIMENT_NAME * "_iter$(iter)" * "_time_$(time)ns" * "_pinq_$(pinqstate)", "pulses/transmon/exptry/")
println("Saving this optimization to $(save_file_path)")
h5open(save_file_path, "cw") do save_file
    for key in keys(result)
        write(save_file, key, result[key])
    end
end



plot_transmon(
    system,
    prob.trajectory,
    plot_path;
    fig_title="X gate on basis states"
)


# data = h5open(save_file_path, "r") do save_file
#     controls = read(save_file, "controls")
#     Δt = read(save_file, "delta_t")
#     return (controls, Δt)
# end

# data
