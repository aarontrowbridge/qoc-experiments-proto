using PicoOld
using LinearAlgebra
using JLD2
using HDF5

const EXPERIMENT_NAME = "MM_X_GATE"

const TRANSMON_LEVELS = 3
const CAVITY_LEVELS = 14
const ketdim = TRANSMON_LEVELS * CAVITY_LEVELS

function cavity_state(level)
    state = zeros(CAVITY_LEVELS)
    state[level + 1] = 1.
    return state
end

#const TRANSMON_ID = I(TRANSMON_LEVELS)

function run_solve(;
    iter = 3000,
    T = 200,
    Δt = 2.5,
    Δt_max = 1.5 * Δt,
    Q = 200.,
    R = 0.001,
    hess = true,
    αval = 20.0,
    ub = 1e-4,
    abfrac = 0.01,
    mode = :free_time,
    resolves = 10,
)


    TRANSMON_G = [1; zeros(TRANSMON_LEVELS - 1)]
    TRANSMON_E = [zeros(1); 1; zeros(TRANSMON_LEVELS - 2)]
    TRANSMON_F = [zeros(2); 1; zeros(TRANSMON_LEVELS - 3)]
    if TRANSMON_LEVELS == 4
        TRANSMON_H = [zeros(3); 1; zeros(TRANSMON_LEVELS - 4)]
    end

    α_ef = -143.277e-3 * 2π
    α_fh = -162.530e-3 * 2π

    A = Diagonal([0., 0., α_ef, α_ef + α_fh])

    χ₂  = -0.63429e-3 * 2π
    χ₂_gf = -1.12885e-3 * 2π
    χ₂_gh = -1.58878e-3 * 2π
    χ₃ = -0.54636e-3 * 2π
    χ₃_gf = -1.017276e-3 * 2π
    χ₃_gh = -1.39180e-3 * 2π

    κ₂ = 5.23e-6 * 2π
    κ₃ = 4.19e-6 * 2π
    κ_cross = 3.6e-6 * 2π

    if TRANSMON_LEVELS == 2
        H_drift = 2*χ₃ * kron(TRANSMON_E*TRANSMON_E', number(CAVITY_LEVELS)) +
                  κ₃/2 * kron(I(TRANSMON_LEVELS), quad(CAVITY_LEVELS))

    elseif TRANSMON_LEVELS == 3
        H_drift = 2*χ₃ * kron(TRANSMON_E*TRANSMON_E', number(CAVITY_LEVELS)) +
                  2*χ₃_gf * kron(TRANSMON_F*TRANSMON_F', number(CAVITY_LEVELS)) +
                  κ₃/2 * kron(I(TRANSMON_LEVELS), quad(CAVITY_LEVELS)) +
                  kron(A[1:TRANSMON_LEVELS, 1:TRANSMON_LEVELS], I(CAVITY_LEVELS))

    elseif TRANSMON_LEVELS == 4
         H_drift = 2*χ₃ * kron(TRANSMON_E*TRANSMON_E', number(CAVITY_LEVELS)) +
                   2*χ₃_gf * kron(TRANSMON_F*TRANSMON_F', number(CAVITY_LEVELS)) +
                   2*χ₃_gh * kron(TRANSMON_H*TRANSMON_H', number(CAVITY_LEVELS)) +
                   κ₃/2 * kron(I(TRANSMON_LEVELS), quad(CAVITY_LEVELS)) +
                   κ_cross* kron(I(TRANSMON_LEVELS), number(CAVITY_LEVELS)) +
                   kron(A[1:TRANSMON_LEVELS, 1:TRANSMON_LEVELS], I(CAVITY_LEVELS))
    else
        error("More than 4 transmon levels not supported yet.")
    end


    # TRANSMON_G = [1; zeros(TRANSMON_LEVELS - 1)]
    # TRANSMON_E = [zeros(1); 1; zeros(TRANSMON_LEVELS - 2)]


    # CHI = 2π * -0.5469e-3
    # KAPPA = 2π * 4e-6

    # H_drift = 2 * CHI * kron(TRANSMON_E*TRANSMON_E', number(CAVITY_LEVELS)) +
    #         (KAPPA/2) * kron(I(TRANSMON_LEVELS), quad(CAVITY_LEVELS))

    transmon_driveR = kron(create(TRANSMON_LEVELS) + annihilate(TRANSMON_LEVELS), I(CAVITY_LEVELS))
    transmon_driveI = kron(1im*(annihilate(TRANSMON_LEVELS) - create(TRANSMON_LEVELS)), I(CAVITY_LEVELS))

    cavity_driveR = kron(I(TRANSMON_LEVELS), create(CAVITY_LEVELS) + annihilate(CAVITY_LEVELS))
    cavity_driveI = kron(I(TRANSMON_LEVELS),  1im * (annihilate(CAVITY_LEVELS) - create(CAVITY_LEVELS)))

    H_drives = [transmon_driveR, transmon_driveI, cavity_driveR, cavity_driveI]

    state0 = kron(TRANSMON_G, cavity_state(0))
    state1 = kron(TRANSMON_G, cavity_state(1))

    ψ1 = [
        state0,
        state1,
        1 / √2 * (state0 - state1),
        1 / √2 * (state0 + 1im * state1)
    ]

    ψf = [
        -1im * state1,
        -1im * state0,
        -1im / √2 * (state1 - state0),
        1 / √2 * (-1im * state1 + state0)
    ]

    # bounds on controls

    # qubit_a_bounds = [0.018 * 2π, 0.018 * 2π]

    # cavity_a_bounds = fill(0.03, 2)

    # a_bounds = [qubit_a_bounds; cavity_a_bounds]



    system = QuantumSystem(
        H_drift,
        H_drives,
        ψ1,
        ψf,
    )

    options = Options(
        max_iter = iter,
        max_cpu_time = 1_000_000.0,
        tol = 1e-6,
        # linear_solver = "pardiso"
    )



    # u_bounds = BoundsConstraint(
    #     1:T,
    #     system.n_wfn_states .+
    #     slice(system.∫a + 1 + system.control_order, system.ncontrols),
    #     ub,
    #     system.vardim
    # )

    # da_bounds = BoundsConstraint(
    #     1:T,
    #     system.n_wfn_states .+ slice(system.∫a + 1 , system.ncontrols),
    #     abfrac.*[qubit_a_bounds; cavity_a_bounds],
    #     system.vardim
    # )

    # top_pop = EqualityConstraint(
    #     1:T,
    #     [[1,2,3,4] .* CAVITY_LEVELS], #[1,2,3,4] .* (CAVITY_LEVELS) .- 1],
    #     0.0,
    #     system.vardim
    # )

    #cons = AbstractConstraint[u_bounds, da_bounds]

    highest_cavity_modes = [
        CAVITY_LEVELS .* [1, 2];
        ketdim .+ (CAVITY_LEVELS .* [1, 2])
    ]

    reg_states = highest_cavity_modes

    if TRANSMON_LEVELS == 3

        transmon_f_states = [
            2 * CAVITY_LEVELS .+ [1:CAVITY_LEVELS...];
            ketdim .+ (2 * CAVITY_LEVELS .+ [1:CAVITY_LEVELS...])
        ]

        append!(reg_states, transmon_f_states)

    elseif TRANSMON_LEVELS == 4

        transmon_h_states = [
            3 * CAVITY_LEVELS .+ [1:CAVITY_LEVELS...];
            ketdim .+ (3 * CAVITY_LEVELS .+ [1:CAVITY_LEVELS...])
        ]

        append!(reg_states, transmon_h_states)
    end



    experiment = "g0_to_g1_transmon_$(TRANSMON_LEVELS)_cavity_$(CAVITY_LEVELS)_" *
        "T_$(T)_dt_max_$(Δt_max)_R_$(R)_iter_$(iter)_ubound_$(ub)"

    plot_dir = "plots/multimode/fermiumL1band"
    data_dir = "data/multimode/fermiumL1band/problems"




    global prob = QuantumControlProblem(
        system;
        T=T,
        Δt=Δt,
        Δt_max=Δt_max,
        Q=Q,
        R=R,
        options=options,
        u_bounds=fill(ub, 4),
        #cons = cons,
        L1_regularized_states=reg_states,
        α = fill(αval, length(reg_states)),
        mode = mode
    )

    for i = 1:resolves

        resolve_experiment = experiment * "_resolve_$i"

        data_file_path = generate_file_path(
            "jld2",
            resolve_experiment,
            data_dir
        )
        plot_path = generate_file_path("png", resolve_experiment, plot_dir)

        plot_multimode_split(prob, plot_path, TRANSMON_LEVELS, CAVITY_LEVELS)
        solve!(prob, save_path = data_file_path)
        plot_multimode_split(prob, plot_path, TRANSMON_LEVELS, CAVITY_LEVELS)

        # raw_controls = jth_order_controls(prob.trajectory, system, 0, d2pi = false)
        # # display(raw_controls)
        # controls = permutedims(reduce(hcat, map(Array, raw_controls)), [2,1])
        # # display(controls)
        # # controls = controls'
        # # display(controls)
        # infidelity = iso_infidelity(final_state(prob.trajectory, system), ket_to_iso(kron(TRANSMON_G, cavity_state(1))))
        g = final_state_i(prob.trajectory, system, 1)
        e = final_state_i(prob.trajectory, system, 2)

        # display(iso_to_ket(g))
        # display(iso_to_ket(e))
        # println(iso_infidelity(g, ket_to_iso(ψf[1])))
        # println(iso_infidelity(e, ket_to_iso(ψf[2])))

        U = hcat(iso_to_ket(g)[1:2], iso_to_ket(e)[1:2])
        Utarg = [0 -im; -im 0]
        println(1/2 * abs(tr(U'Utarg)))

        # result = Dict(
        #     "total_time" => T * Δt,
        #     "T" => T,
        #     "delta_t" => Δt,
        #     # "trajectory" => prob.trajectory,
        #     #"a_max" => amax,
        #     "controls" => controls,
        #     "infidelity" => infidelity
        # )


        controls_file_path = generate_file_path("h5", resolve_experiment, "pulses/multimode/fermiumL1band/")

        println("Saving this optimization to $(controls_file_path)")

        get_and_save_controls(data_file_path, controls_file_path)

        # h5open(save_file_path, "cw") do save_file
        #     for key in keys(result)
        #         write(save_file, key, result[key])
        #     end
        # end

        global prob = load_problem(data_file_path)
    end
end

run_solve()
