using Pico
using LinearAlgebra
using JLD2
using HDF5



const EXPERIMENT_NAME = "g0_to_g1"

const TRANSMON_LEVELS = 2
const CAVITY_LEVELS = 4

function cavity_state(level)
    state = zeros(CAVITY_LEVELS)
    state[level + 1] = 1.
    return state
end
#const TRANSMON_ID = I(TRANSMON_LEVELS)

function run_solve(;iter = 3000, T = 1100,  Δt = 3., Q = 200., R = 0.1, hess = true, pinqstate = false, αval = 0.25, ub = 0.0002)

    
    TRANSMON_G = [1; zeros(TRANSMON_LEVELS - 1)]
    TRANSMON_E = [zeros(1); 1; zeros(TRANSMON_LEVELS - 2)]


    CHI = 2π * -0.5469e-3
    KAPPA = 2π * 4e-6

    H_drift = 2 * CHI * kron(TRANSMON_E*TRANSMON_E', number(CAVITY_LEVELS)) +
            (KAPPA/2) * kron(I(TRANSMON_LEVELS), quad(CAVITY_LEVELS))

    transmon_driveR = kron(create(TRANSMON_LEVELS) + annihilate(TRANSMON_LEVELS), I(CAVITY_LEVELS))
    transmon_driveI = kron(1im*(annihilate(TRANSMON_LEVELS) - create(TRANSMON_LEVELS)), I(CAVITY_LEVELS))

    cavity_driveR = kron(I(TRANSMON_LEVELS), create(CAVITY_LEVELS) + annihilate(CAVITY_LEVELS))
    cavity_driveI = kron(I(TRANSMON_LEVELS),  1im * (annihilate(CAVITY_LEVELS) - create(CAVITY_LEVELS)))

    H_drives = [transmon_driveR, transmon_driveI, cavity_driveR, cavity_driveI]

    ψ1 = kron(TRANSMON_G, cavity_state(0))
    ψf = kron(TRANSMON_G, cavity_state(1))

    # bounds on controls

    qubit_a_bounds = [0.018 * 2π, 0.018 * 2π]

    cavity_a_bounds = fill(0.03, 2)

    a_bounds = [qubit_a_bounds; cavity_a_bounds]


    pin_first_qstate = pinqstate

    system = QuantumSystem(
        H_drift,
        H_drives,
        ψ1,
        ψf,
        a_bounds,
    )

    options = Options(
        max_iter = iter,
        max_cpu_time = 80000.0,
        tol = 1e-6
    )



    u_bounds = BoundsConstraint(
        1:T,
        system.n_wfn_states .+
        slice(system.∫a + 1 + system.control_order, system.ncontrols),
        ub,
        system.vardim
    )

    top_pop = EqualityConstraint(
        1:T,
        [[1,2,3,4] .* CAVITY_LEVELS; [1,2,3,4] .* (CAVITY_LEVELS) .- 1],
        0.0,
        system.vardim
    )

    cons = AbstractConstraint[u_bounds, top_pop]

    experiment = "g0_to_g1_T_$(T)_dt_$(Δt)_R_$(R)_iter_$(iter)" * (pin_first_qstate ? "_pinned" : "") * "ubound_$(ub)"

    plot_dir = "plots/multimode/fermiumL1band"
    data_dir = "data/multimode/fixed_time/no_guess/problems"

    plot_path = generate_file_path("png", experiment, "plots/multimode/fermiumL1band/")



    prob = QuantumControlProblem(
        system;
        T=T,
        Δt=Δt,
        Q = Q,
        R=R,
        pin_first_qstate=pin_first_qstate,
        options=options,
        cons = cons
        # L1_regularized_states=[1,2,3,4] .* CAVITY_LEVELS,
        # α = fill(αval, 4)
    )
    data_path = generate_file_path(
            "jld2",
            experiment,
            data_dir
        )

    plot_multimode_split(prob, plot_path)
    solve!(prob, save_path = data_path)
    plot_multimode_split(prob, plot_path)

    raw_controls = jth_order_controls(prob.trajectory, system, 0, d2pi = false)
    display(raw_controls)
    controls = permutedims(reduce(hcat, map(Array, raw_controls)), [2,1])
    display(controls)
    # controls = controls'
    # display(controls)
    infidelity = iso_infidelity(final_state(prob.trajectory, system), ket_to_iso(kron(TRANSMON_G, cavity_state(1))))
    print(infidelity)

    result = Dict(
    "total_time" => T * Δt,
    "T" => T,
    "delta_t" => Δt,
    # "trajectory" => prob.trajectory,
    #"a_max" => amax,
    "controls" => controls,
    "infidelity" => infidelity
    )


    save_file_path = generate_file_path("h5", experiment, "pulses/multimode/fermiumL1band/")
    println("Saving this optimization to $(save_file_path)")
    h5open(save_file_path, "cw") do save_file
        for key in keys(result)
            write(save_file, key, result[key])
        end
    end


end
# let sol = true, i = 0

#     while sol
#         resolve = "_resolve_$(i)"
#         plot_path = generate_file_path(
#             "png",
#             experiment * resolve,
#             plot_dir
#         )
#         data_path = generate_file_path(
#             "jld2",
#             experiment * resolve,
#             data_dir
#         )
#         plot_multimode(system, prob.trajectory, plot_path)
#         solve!(prob, save=true, path=data_path)
#         plot_multimode(system, prob.trajectory, plot_path)

#         raw_controls = jth_order_controls(prob.trajectory, system, 0, d2pi = false)
#         display(raw_controls)
#         controls = permutedims(reduce(hcat, map(Array, raw_controls)), [2,1])
#         display(controls)
#         infidelity = iso_infidelity(final_state(prob.trajectory, system), ket_to_iso(kron(TRANSMON_G, cavity_state(1))))
#         print(infidelity)

#         result = Dict(
#         "total_time" => T * Δt,
#         "T" => T,
#         "delta_t" => Δt,
#         # "trajectory" => prob.trajectory,
#         #"a_max" => amax,
#         "controls" => controls,
#         "infidelity" => infidelity
#         )


#         save_file_path = generate_file_path("h5", experiment, "pulses/multimode/fastfmcomp")
#         println("Saving this optimization to $(save_file_path)")
#         h5open(save_file_path, "cw") do save_file
#             for key in keys(result)
#                 write(save_file, key, result[key])
#             end
#         end
        
        
#         prompt = false
#         while prompt 
#             println("Resolve? (y/n)")
#             answer = readline()
#             if answer == "y"
#                 global prob = load_object(data_path)
#                 prompt = false
#             elseif answer == "n"
#                 prompt = false
#                 sol = false
#             else 
#                 println("Invalid response, must be y or n")
#             end
#         end
#         i +=1
#         sol = false
#     end
# end
