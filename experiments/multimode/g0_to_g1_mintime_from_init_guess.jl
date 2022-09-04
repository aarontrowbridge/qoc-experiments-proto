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

mintime_iter = parse(Int, ARGS[2])

R = parse(Float64, ARGS[3])

R_mintime = parse(Float64, ARGS[4])

pin_first_qstate = parse(Bool, ARGS[5])

save_controls = parse(Bool, ARGS[6])


# script parameters

options = Options(
    max_iter = iter,
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

mintime_options = Options(
    max_iter = mintime_iter,
    max_cpu_time = 100000.0,
)

mintime_prob = MinTimeProblem(
    prob;
    mintime_options=mintime_options,
    R=R,
    options=options,
    Rᵤ=R_mintime,
    a_bound=a_bounds,
    pin_first_qstate=pin_first_qstate
)

# define plot location

plot_path = "plots/multimode/mintime/g0_to_g1_init_traj_T_$(T)_R_$(R)_Ru_$(R_mintime)_iter_$(iter)_mintime_iter_$(mintime_iter).png"


# test the plot

plot_multimode(system, prob.trajectory, plot_path)


# solve multimode problem

solve!(mintime_prob)


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
