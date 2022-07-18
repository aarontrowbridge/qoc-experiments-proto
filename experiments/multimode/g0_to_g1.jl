using QubitControl
using HDF5

hf_path = "notebooks/g0_to_g1_multimode_system_data.h5"

hf = h5open(hf_path, "r")

H_drift = copy(transpose(hf["H_drift"][:, :]))
H_drives = [copy(transpose(hf["H_drives"][:, :, i])) for i = 1:size(hf["H_drives"], 3)]

ψ1 = vcat(transpose(hf["psi1"][:, :])...)
ψf = vcat(transpose(hf["psif"][:, :])...)

ts = hf["tlist"][:]

system = MultiModeQubitSystem(
    H_drift,
    H_drives,
    ψ1,
    ψf
)

Δt = ts[2] - ts[1]

controls = copy(transpose(hf["controls"][:,:]))

init_traj = Trajectory(system, controls, Δt)

iter = parse(Int, ARGS[1])

options = Options(
    max_iter = iter,
    tol = 1.0e-8,
    max_cpu_time = 100000.0,
    linear_solver = "mumps"
)

Δt = 0.5
T = 1000

pin_first_qstate = false

prob = QubitProblem(
    system,
    T;
    # init_traj;
    Δt=Δt,
    pin_first_qstate=pin_first_qstate,
    options=options
)

plot_path = "plots/multimode/g0_to_g1_iter_$(iter)_dt_$(Δt)_T_$(T).png"

plot_multimode_qubit(system, prob.trajectory, plot_path)

solve!(prob)

pico_controls = controls_matrix(prob.trajectory, prob.system)

h5open("notebooks/controls_data.h5", "w") do hdf
    hdf["controls"] = pico_controls
end

plot_multimode_qubit(system, prob.trajectory, plot_path)
