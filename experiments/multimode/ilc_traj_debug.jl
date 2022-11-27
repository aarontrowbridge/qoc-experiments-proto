using Pico

# data_dir = "data/multimode/fixed_time_update/guess/pinned/problems"
data_dir = "experiments/multimode"

# data_name = "g0_to_g1_T_102_dt_4.0_Q_500.0_R_0.1_iter_2000_u_bound_1.0e-5_alpha_transmon_20.0_alpha_cavity_20.0_resolve_5_00000"

data_name = "g0_to_g1_T_101_dt_4.0_Q_500.0_R_0.1_u_bound_1.0e-5"

data_path = joinpath(data_dir, data_name * ".jld2")

data = load_data(data_path)

xs = [
    data.trajectory.states[t][end - 7: end]
        for t = 1:data.trajectory.T
]


us = [
    data.trajectory.actions[t] 
        for t = 1:data.trajectory.T
]

Δt = data.trajectory.Δt
T = data.trajectory.T

for i = 1:T-1
    us[i] = us[i+1]
end

#xs[1][end - 3:end] .= Δt * us[1]

function integrate_controls(us, Δt, T)
    int_controls = [fill(0., 8) for t in 1:T]


    #int_controls[1][end-3:end] = xs[2][end - 3:end]

    for t = 2:T
        int_controls[t][end - 3: end] = int_controls[t-1][end-3:end] + Δt * us[t-1]
        int_controls[t][end - 7: end - 4] = int_controls[t-1][end - 7: end - 4] + Δt*int_controls[t][end - 3: end]
        #int_controls[t] = int_controls[t-1] + [int_controls[t-1][end - 3 : end]; us[t-1]] * Δt 
    end
    return int_controls
end


display((integrate_controls(us, Δt, T) - xs)[[1:5; end - 5: end]])
println("Integrate Controls")
display(integrate_controls(us, Δt, T)[[1:5; end - 4:end]])
println("PICO controls")
display(xs[[1:5; end - 4: end]])
