WDIR = joinpath(@__DIR__, "../../")
using Pico
using HDF5

hf_path = "notebooks/g0_to_g1_multimode_system_data.h5"

system, data =
    QuantumSystem(hf_path; return_data=true)

Δt = data["Δt"]
T = data["T"]


# parsed parameters

iter = parse(Int, ARGS[1])

R = parse(Float64, ARGS[2])

pin_first_qstate = parse(Bool, ARGS[3])

save_controls = parse(Bool, ARGS[4])


# script parameters

tol = 1e-8

options = Options(
    max_iter = iter,
    tol = tol,
    max_cpu_time = 100000.0,
)


# bounds on controls

qubit_a_bounds = [0.018 * 2π, 0.018 * 2π]

cavity_a_bounds = fill(0.03, system.ncontrols - 2)

a_bounds = [qubit_a_bounds; cavity_a_bounds]


# create initial trajectory

init_traj = Trajectory(system, data["controls"], Δt)


# define problem

prob = QuantumControlProblem(
    system,
    init_traj;
    R=R,
    pin_first_qstate=pin_first_qstate,
    options=options
)


# define plot location

plot_path = "plots/multimode/fixed_time/guess/g0_to_g1_init_traj_T_$(T)_R_$(R)_iter_$(iter)" * (pin_first_qstate ? "_pinned" : "") * ".png"


# test the plot

plot_multimode(system, prob.trajectory, plot_path)


# solve multimode problem

solve!(prob)


# plot results

plot_multimode(system, prob.trajectory, plot_path)


# save results if requested

if save_controls
    pico_controls = controls_matrix(prob.trajectory, prob.system)
    tlist = hcat(prob.trajectory.times...)

    h5open("notebooks/controls_data.h5", "w") do hdf
        hdf["controls"] = pico_controls
        hdf["tlist"] = tlist
    end
end
