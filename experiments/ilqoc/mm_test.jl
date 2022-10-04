using Pico

WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "experiments", "ilqoc", "wigner.jl"))

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
    return G_drift + a[1]*G_drives[1] + a[2]*G_drives[2] + a[3]*G_drives[3] + a[4]*G_drives[4]
end

TRANSMON_G = [1; zeros(2 - 1)]
TRANSMON_E = [zeros(1); 1; zeros(2 - 2)]
CHI = 2π * -0.5469e-3

sys = prob.system
Δt = prob.trajectory.Δt

function g2(x, sys::QuantumSystem)
    y = []
    #ψiso = x[slice(1, sys.isodim)]
    cavity_levels = (sys.isodim ÷ 2) ÷ 2
    #ψ = iso_to_ket(ψiso)
    for i = 1:2
        append!(y, x[i]^2 + x[i+2*cavity_levels]^2)
        append!(y, x[i + cavity_levels]^2 + x[i+3*cavity_levels]^2)
    end
    append!(y, sum(x[1:cavity_levels].^2 + x[2*cavity_levels .+ (1:cavity_levels)].^2))
    append!(y, sum(x[cavity_levels .+ (1:cavity_levels)].^2 + x[3*cavity_levels .+ (1:cavity_levels)].^2))
    return y 
end

function g_wig(x, sys::QuantumSystem)
    y = []
    alphas = [0. + 0*im, -0.46056592 + 0.96496662*im, -1.12217844 + 0.05909296*im,
    1.15574006 + 0.00121442*im, -0.31511958 - 1.07263407*im,  0.03383244 + 0.03960534*im,
    0.48478839 + 0.35503407*im,  0.78252667 - 0.84001832*im,  0.00502949 + 1.10619208*im,
    -0.53343445 + 0.04504328*im, 0.79600605 + 0.77434995*im, -0.357696 - 0.94484753*im,
    -0.93217126 + 0.66051767*im,  0.67539941 - 0.75525661*im, -0.93395687 - 0.31345291*im,
    -0.14257336 - 0.49161566*im,  0.48699864 - 0.29529878*im,  0.99365245 + 0.55117771*im,
    -0.14431765 + 0.56553579*im]

   for α in alphas 
       append!(y, meas_W_iso(x[1:sys.n_wfn_states], α, Float64(π), cavity_levels))
   end
   return y
end




function g(x, sys::QuantumSystem)
    return x[1:sys.n_wfn_states]
end

function exp_rollout(utraj::Matrix{Float64})
    ys = []
    ts = []
    cavity_levels = (sys.isodim ÷ 2) ÷ 2
    state = [ket_to_iso(kron(TRANSMON_G, cavity_state(0))); zeros(8)]
    G_drift = sys.G_drift + get_mat_iso(-1im*2 * CHI*0.05 * kron(TRANSMON_E*TRANSMON_E', number(cavity_levels)))
    for k = 1:size(utraj,2)
        G = gen_G(state[sys.isodim .+ (1:4)], G_drift, sys.G_drives)
        h_prop = exp(G * Δt)
        state_ = h_prop*state[1:sys.isodim]
        controls = state[sys.isodim .+ (1:4)] + state[sys.isodim .+ (5:8)] .* Δt
        dcontrols = state[sys.isodim .+ (5:8)] + utraj[:, k] .* Δt
        state = [state_; controls; dcontrols]
        append!(ys, [g2(state, sys)])
        append!(ts, k+1)
    end
    return [ys[end ÷ 2], ys[end]], [ts[end ÷2], ts[end]]
end
 
ilc_prob = ILCProblem(prob, g2, exp_rollout, 6; solve = false)
#println(g2(ilc_prob.ỹs[end], sys))
#y1, t1s = exp_rollout(ilc_prob.utraj)
#println(g2(y1[end], sys))
answer, ts = solve_ilc!(ilc_prob; iter = 3)
println(answer[end])

#println(g2(answer[end], sys))

