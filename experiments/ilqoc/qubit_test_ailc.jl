using PicoOld

WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "experiments", "ilqoc", "measurement.jl"))




#system parameters

ω = 2π * 4.96 #GHz
α = -2π * 0.143 #GHz
levels = 2

ψg = [1. + 0*im, 0. ]
ψe = [0., 1. + 0*im]

ψ1 = [ψg, ψe]
ψf = [-im*ψe, -im*ψg]

H_drift = α/2 * quad(levels)
H_drive = [create(levels) + annihilate(levels),
1im * (create(levels) - annihilate(levels))]

control_bounds = [2π * 19e-3,  2π * 19e-3]

system = QuantumSystem(
    H_drift,
    H_drive,
    ψ1,
    ψf,
    control_bounds
)

#T is number of time steps, not total time
T = 400
Δt = 0.1
Q = 200.
R = 0.1
cost = :infidelity_cost
hess = true
pinqstate = true

time = T * Δt

options = Options(
    max_iter = 500,
    tol = 1e-5
)

prob = QuantumControlProblem(
    system;
    T=T,
    Δt = Δt,
    Q = Q,
    R = R,
    eval_hessian = hess,
    cost = cost,
    pin_first_qstate = pinqstate,
    options = options
)

solve!(prob)

xs = [
    prob.trajectory.states[t]
        for t = 1:prob.trajectory.T
]

us = [
    prob.trajectory.actions[t]
        for t = 1:prob.trajectory.T
]


Ẑ = Trajectory(
    xs,
    us,
    prob.trajectory.times,
    prob.trajectory.T,
    prob.trajectory.Δt
)

function g(x) #sys::QuantumSystem)
    #x[1:sys.n_wfn_states]
    y = []
    for i = 1:system.nqstates
        ψ_i = x[slice(i, system.isodim)]
        append!(y, meas_x_iso(ψ_i))
        append!(y, meas_y_iso(ψ_i))
        append!(y, meas_z_iso(ψ_i))
    end
    return y
end

experiment = QuantumExperiment(
    prob.system,
    Ẑ.states[1],
    Ẑ.Δt,
    g,
    3*prob.system.nqstates,
    1:Ẑ.T
)

prob = ILCProblem(
    prob.system,
    Ẑ,
    experiment;
    max_iter = 10
)

solve!(prob)
