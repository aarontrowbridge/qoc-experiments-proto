# WDIR = joinpath(@__DIR__, "../../")
using PicoOld
using HDF5

hf_path = "notebooks/g0_to_g1_multimode_system_data.h5"

system, data = QuantumSystem(hf_path; return_data=true)


R                = 1.0
iter             = 100
pin_first_qstate = false
save_controls    = true

# R                = parse(Float64, ARGS[2])
# iter             = parse(Int, ARGS[1])
# pin_first_qstate = parse(Bool, ARGS[3])
# save_controls    = parse(Bool, ARGS[4])


# create initial trajectory

init_traj = Trajectory(system, data["controls"], data["Δt"])


# define problem

prob = QuantumControlProblem(
    system,
    init_traj;
    R=R,
    pin_first_qstate=pin_first_qstate,
    options=Options(
        max_iter=iter,
    ),
)


# define plot location

plot_path = "plots/multimode/fixed_time/guess/g0_to_g1_init_traj_R_$(R)_iter_$(iter)" * (pin_first_qstate ? "_pinned" : "") * ".png"


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
