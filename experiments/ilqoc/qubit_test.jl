using Pico

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

function g(x, sys::QuantumSystem)
    #x[1:sys.n_wfn_states]
    y = []
    for i = 1:sys.nqstates
        ψ_i = x[slice(i, sys.isodim)]
        append!(y, meas_x_iso(ψ_i))
        append!(y, meas_y_iso(ψ_i))
        append!(y, meas_z_iso(ψ_i))
    end
    return y 
end

function exp_rollout(utraj::Matrix{Float64})
    state = [ket_to_iso(ψg); ket_to_iso(ψe); zeros(4)]
    ys = []
    ts = []
    for k in 1:size(utraj,2)
        G = system.G_drift + 0.01*get_mat_iso(-1im * sigmaz()) + state[end - 3] * system.G_drives[1] + state[end-2] * system.G_drives[2]
        h_prop = exp(G * Δt)
        state1_ = h_prop*state[1:4]
        state2_ = h_prop*state[5:8]
        controls = state[9:10] + state[11:12] .* Δt
        dcontrols = state[11:12] + utraj[:, k] .* Δt
        state = [state1_; state2_; controls; dcontrols]
        append!(ys, [g(state, system)])
        append!(ts, k+1)
    end
    return ys, ts
end

ilc_prob = ILCProblem(prob, g, exp_rollout, system.n_wfn_states)
answer, jku = solve_ilc!(ilc_prob; iter = 1)

println(answer)