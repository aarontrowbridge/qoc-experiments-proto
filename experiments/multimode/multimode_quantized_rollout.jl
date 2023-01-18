using Pico
import Pico.Trajectories: rollout
using Makie
using HDF5

# data_path = "data/multimode/free_time/no_guess/good_solutions/g0_to_g1_T_100_dt_10.0_dt_max_factor_1.0_Q_1000.0_R_1.0e-5_iter_2000_u_bound_1.0e-6_alpha_transmon_20.0_alpha_cavity_20.0_resolve_5_00000.jld2"

data_path = "data/multimode/free_time/no_guess/problems/g0_to_g1_T_100_dt_10.0_Δt_max_factor_1.0_Q_1000.0_R_1.0e-5_iter_2000_u_bound_1.0e-5_alpha_transmon_20.0_alpha_cavity_20.0_resolve_5_00000.jld2"

data_path = "data/multimode/free_time/no_guess/problems/g0_to_g1_T_100_dt_4.0_Δt_max_factor_2.0_Q_1000.0_R_1.0e-5_iter_2000_u_bound_1.0e-5_alpha_transmon_20.0_alpha_cavity_20.0_resolve_3_00000.jld2"

data_path = "data/multimode/free_time/no_guess/problems/g0_to_g1_T_100_dt_4.0_Δt_max_factor_2.0_Q_1000.0_R_1.0e-5_iter_2000_u_bound_1.0e-5_alpha_transmon_20.0_alpha_cavity_20.0_resolve_9_00001.jld2"

aditya_pulse = "data/multimode/from_aditya/g0_to_g1_T_200_dt_2.0_R_0.1_iter_3000ubound_0.001_00000.h5"

# load aditya_pulse
file = h5open(aditya_pulse, "r")

A = read(file, "controls") |> transpose
dt = read(file, "delta_t")[1]
T = read(file, "T")[1]


as_aditya = [A[:, t] for t ∈ axes(A, 2)]

T
dt
as

dts = [dt for t = 1:T]

prob = load_data(data_path)

as = [prob.trajectory.states[t][prob.system.n_wfn_states .+ slice(1, prob.system.ncontrols)] for t = 1:prob.trajectory.T]

fidelity(ψ̃, ψ̃goal) = abs2(iso_to_ket(ψ̃)' * iso_to_ket(ψ̃goal))

a_bounds = prob.system.a_bounds

# quantize controls with 8 bit precision
function quantize(as::Vector{Vector{Float64}})
    bs = [255 ./ a_bounds .* (a .+ a_bounds) ./ 2 for a in as]
    cs = [round.(b) for b in bs]
    return [a_bounds / 128 .* (c .- 128) for c in cs]
end


qubit_res = a_bounds[1] / 128
cavity_res = a_bounds[3] / 128

function quantize2(as::Vector{Vector{Float64}})
    res = a_bounds ./ 128
    return [floor.(a ./ res) .* res for a in as]
end

quantized_controls = quantize(as)
series(hcat(quantized_controls...))

quantized_controls2 = quantize2(as)
series(hcat(quantized_controls2...))

as_aditya

quantized_controls_aditya = quantize(as_aditya)
series(hcat(quantized_controls_aditya...))

quantized_controls_aditya2 = quantize2(as_aditya)
fig, ax, = series(hcat(quantized_controls_aditya2...))
axislegend(ax)
fig


prob.system.a_bounds
series(hcat(as...))


Δt = [
    prob.trajectory.times[t + 1] - prob.trajectory.times[t]
        for t = 1:prob.trajectory.T - 1
]

Ψ̃ = rollout(prob.system, as, Δt)
Ψ̃_quantized = rollout(prob.system, quantized_controls, Δt)
Ψ̃_quantized2 = rollout(prob.system, quantized_controls2, Δt)


Ψ̃_aditya_quantized = rollout(prob.system, quantized_controls_aditya, dts)
Ψ̃_aditya_quantized2 = rollout(prob.system, quantized_controls_aditya2, dts)
Ψ̃_aditya = rollout(prob.system, as_aditya, dts)


i = slice(1, prob.system.isodim)

# smooth controls fidelity
fidelity(Ψ̃[end][i], prob.system.ψ̃goal[i])

# quantized controls fidelity
fidelity(Ψ̃_quantized[end][i], prob.system.ψ̃goal[i])

# quantized controls fidelity
fidelity(Ψ̃_quantized2[end][i], prob.system.ψ̃goal[i])


# smooth aditya controls fidelity
fidelity(Ψ̃_aditya[end][i], prob.system.ψ̃goal[i])

# quantized aditya controls fidelity
fidelity(Ψ̃_aditya_quantized[end][i], prob.system.ψ̃goal[i])

# quantized aditya controls fidelity
fidelity(Ψ̃_aditya_quantized2[end][i], prob.system.ψ̃goal[i])
