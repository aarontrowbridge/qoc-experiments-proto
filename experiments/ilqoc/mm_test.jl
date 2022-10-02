using Pico

WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "experiments", "ilqoc", "exponential.jl"))

prob = load_problem("experiments/ilqoc/probdata/g0_to_g1_T_581_dt_4.0_R_0.1_iter_3000ubound_0.0001_00001.jld2")

sys = prob.system
cavity_levels = (sys.isodim ÷ 2) ÷ 2

function cavity_state(level)
    state = zeros(cavity_levels)
    state[level + 1] = 1.
    return state
end

function gen_G(
    a::AbstractVector,
    G_drift::AbstractMatrix,
    G_drives::AbstractVector{<:AbstractMatrix}
)
    return G_drift + sum(a .* G_drives)
end

TRANSMON_G = [1; zeros(2 - 1)]
TRANSMON_E = [zeros(1); 1; zeros(2 - 2)]
CHI = 2π * -0.5469e-3

sys = prob.system
Δt = prob.trajectory.Δt

function g(x, sys::QuantumSystem)
    y = []
    ψiso = x[slice(1, sys.isodim)]
    cavity_levels = (sys.isodim ÷ 2) ÷ 2
    ψ = iso_to_ket(ψiso)
    for i = 1:8
        append!(y, abs2(ψ[i]) + abs2(ψ[i + cavity_levels]))
    end
    append!(y, sum(abs2.(ψ[1:cavity_levels])))
    append!(y, sum(abs2.(ψ[cavity_levels + 1:end])))
end

function exp_rollout(utraj::Matrix{Float64})
    ys = []
    ts = []
    cavity_levels = (sys.isodim ÷ 2) ÷ 2
    state = [ket_to_iso(kron(TRANSMON_G, cavity_state(0))); zeros(8)]
    G_drift = sys.G_drift + get_mat_iso(-1im*2 * CHI*0.1 * kron(TRANSMON_E*TRANSMON_E', number(cavity_levels)))
    for k = 1:size(utraj,2)
        G = gen_G(state[sys.isodim .+ (1:4)], G_drift, sys.G_drives)
        h_prop = exp(G * Δt)
        state_ = h_prop*state[1:sys.isodim]
        controls = state[sys.isodim .+ (1:4)] + state[sys.isodim .+ (5:8)] .* Δt
        dcontrols = state[sys.isodim .+ (5:8)] + utraj[:, k] .* Δt
        state = [state_; controls; dcontrols]
        append!(ys, [g(state, sys)])
        append!(ts, k+1)
    end
    return ys, ts
end
 
ilc_prob = ILCProblem(prob, g, exp_rollout, 10; solve = false)
println(ilc_prob.ỹs[end])
answer, ts = solve_ilc!(ilc_prob; iter = 2)

println(answer[end])

