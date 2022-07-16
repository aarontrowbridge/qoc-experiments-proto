using QubitControl
using HDF5

hf_path = "notebooks/g0_to_g1_multimode_system_data.h5"

hf = h5open(hf_path, "r")

H_drift = hf["H_drift"][:, :]
H_drives = [hf["H_drives"][:, :, i] for i = 1:size(hf["H_drives"], 3)]

ψ1 = vcat(transpose(hf["psi1"][:, :])...)
ψf = vcat(transpose(hf["psif"][:, :])...)

ts = hf["tlist"][:]

system = MultiModeQubitSystem(
    H_drift,
    H_drives,
    ψ1,
    ψf,
    ts
)

Δt = ts[2] - ts[1]

controls = copy(transpose(hf["controls"][:,:]))

init_traj = TrajectoryData(system, Δt, controls)

iter = parse(Int, ARGS[1])

options = Options(
    max_iter = iter,
    tol = 1.0e-8,
    max_cpu_time = 5000.0
)

Δt = 0.1
T = 1000

pin_first_qstate = false


prob = QubitProblem(
    system,
    T;
    Δt=Δt,
    pin_first_qstate=pin_first_qstate,
    options=options
)

plot_path = "plots/multimode/g0_to_g1_iter_$(iter)_dt_$(Δt)_T_$(T).png"

plot_multimode_qubit(system, prob.trajectory, plot_path)

solve!(prob)

pico_controls = hcat(
    [
        [
            prob.trajectory.states[t][
                prob.system.n_wfn_states + index(k, 2, prob.system.augdim)
            ]
            for k = 1:prob.system.ncontrols
        ] for t = 1:prob.T
    ]...
)

h5open("notebooks/controls_data.h5", "w") do hdf
    hdf["controls"] = pico_controls
end

plot_multimode_qubit(system, prob.trajectory, plot_path)
