WDIR = joinpath(@__DIR__, "../../")
using QubitControl
using HDF5

hf_path = "notebooks/g0_to_g1_multimode_system_data.h5"

hf = h5open(hf_path, "r")

print(keys(hf))

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

Δt_init_traj = ts[2] - ts[1]

controls = copy(transpose(hf["controls"][:,:]))

init_traj = Trajectory(system, controls, Δt_init_traj)

iter = parse(Int, ARGS[4])

options = Options(
    max_iter = iter,
    tol = 1.0e-8,
    max_cpu_time = 100000.0,
    linear_solver = "mumps"
)

T = parse(Int, ARGS[1])
Δt = parse(Float64, ARGS[2])
R = parse(Float64, ARGS[3])

traj_guess = parse(Bool, ARGS[5])

if traj_guess
    Δt = Δt_init_traj
    T = length(ts)
end

pin_first_qstate = parse(Bool, ARGS[6])

if traj_guess
    prob = QubitProblem(
        system,
        init_traj;
        Δt=Δt,
        R=R,
        pin_first_qstate=pin_first_qstate,
        options=options
    )
else
    prob = QubitProblem(
        system,
        T;
        Δt=Δt,
        R=R,
        pin_first_qstate=pin_first_qstate,
        options=options
    )
end

plot_path = "plots/multimode/g0_to_g1_T_$(T)_dt_$(Δt)_R_$(R)_iter_$(iter)" *
    (traj_guess ? "_init_traj" : "") * (pin_first_qstate ? "_pinned" : "") * ".png"

plot_multimode_qubit(system, prob.trajectory, plot_path)

solve!(prob)

pico_controls = controls_matrix(prob.trajectory, prob.system)
tlist = hcat(prob.trajectory.times...)

h5open("notebooks/controls_data.h5", "w") do hdf
    hdf["controls"] = pico_controls
    hdf["tlist"] = tlist
end

plot_multimode_qubit(system, prob.trajectory, plot_path)
