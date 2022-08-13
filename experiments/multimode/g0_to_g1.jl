WDIR = joinpath(@__DIR__, "../../")
using QubitControl
using HDF5

hf_path = "notebooks/g0_to_g1_multimode_system_data.h5"

system = QuantumSystem(hf_path)

T = parse(Int, ARGS[1])
Δt = parse(Float64, ARGS[2])
R = parse(Float64, ARGS[3])
iter = parse(Int, ARGS[4])
pin_first_qstate = parse(Bool, ARGS[5])
phase_flip = parse(Bool, ARGS[6])

options = Options(
    max_iter = iter,
    max_cpu_time = 100000.0,
)

# bounds on controls

qubit_a_bounds = [0.018 * 2π, 0.018 * 2π]

cavity_a_bounds = fill(0.03, system.ncontrols - 2)

a_bounds = [qubit_a_bounds; cavity_a_bounds]

prob = QuantumControlProblem(
    system,
    T;
    Δt=Δt,
    R=R,
    a_bound=a_bounds,
    phase_flip=phase_flip,
    pin_first_qstate=pin_first_qstate,
    options=options
)

plot_path = "plots/multimode/fixed_time/no_guess/g0_to_g1_T_$(T)_dt_$(Δt)_R_$(R)_iter_$(iter)" * (pin_first_qstate ? "_pinned" : "") * (phase_flip ? "_phase_flip" : "") * ".png"

plot_multimode_qubit(system, prob.trajectory, plot_path)

solve!(prob)

# pico_controls = controls_matrix(prob.trajectory, prob.system)
# tlist = hcat(prob.trajectory.times...)

# h5open("notebooks/controls_data.h5", "w") do hdf
#     hdf["controls"] = pico_controls
#     hdf["tlist"] = tlist
# end

plot_multimode_qubit(system, prob.trajectory, plot_path)
