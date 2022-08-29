# WDIR = joinpath(@__DIR__, "../../")
using Pico
using HDF5

hf_path = "notebooks/g0_to_g1_multimode_sys_data.h5"

system = QuantumSystem(hf_path, control_bounds = a_bounds)
# bounds on controls

qubit_a_bounds = [0.018 * 2π, 0.018 * 2π]

cavity_a_bounds = fill(0.03, sys.ncontrols - 2)

a_bounds = [qubit_a_bounds; cavity_a_bounds]

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

u_bounds = BoundsConstraint(
    1:T,
    sys.n_wfn_states .+
    slice(sys.∫a + 1 + sys.control_order, sys.ncontrols),
    0.0001,
    sys.vardim
)

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

plot_multimode(sys, prob.trajectory, plot_path)

save_path = "data/multimode/fixed_time/no_guess/g0_to_g1_T_$(T)_dt_$(Δt)_R_$(R)_iter_$(iter)" * (pin_first_qstate ? "_pinned" : "") * (phase_flip ? "_phase_flip" : "") * ".h5"

solve!(prob, save=true, path=save_path)

plot_multimode(sys, prob.trajectory, plot_path)
