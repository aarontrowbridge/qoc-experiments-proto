using Pico
using Test

using ForwardDiff
using FiniteDiff
using SparseArrays
using LinearAlgebra


#
# setting up simple quantum system
#

σx = GATES[:X]
σy = GATES[:Y]
σz = GATES[:Z]

H_drift = σz / 2
H_drive = [σx / 2, σy / 2]

gate = :X

ψ0 = [1, 0]
ψ1 = [0, 1]

# ψ = [ψ0, ψ1, (ψ0 + im * ψ1) / √2, (ψ0 - ψ1) / √2]
ψ = [ψ0, ψ1]
ψf = apply.(gate, ψ)

system = QuantumSystem(
    H_drift,
    H_drive,
    ψ,
    ψf,
    [1.0, 0.5]
)



"""
    Testing derivatives

"""


T = 5

@assert T > 4 "mintime objective Hessian is set up for T > 4"

Q = 200.0
R = 1.0
Rᵤ = 1.0
Rₛ = 1.0



eval_hessian = true

cost_fn = :infidelity_cost


# absolulte tolerance for approximate tests

const ATOL = 1e-4







#
# helper functions
#


# convert sparse data to dense matrix

function dense(vals, structure, shape)

    M = zeros(shape)

    for (v, (k, j)) in zip(vals, structure)
        M[k, j] += v
    end

    if shape[1] == shape[2]
        return Symmetric(M)
    else
        return M
    end
end


# show differences between arrays

function show_diffs(A, B)
    for (i, (a, b)) in enumerate(zip(A, B))
        inds = Tuple(CartesianIndices(A)[i])
        if !isapprox(a, b, atol=ATOL) && inds[1] ≤ inds[2]
            println((a, b), " @ ", inds)
        end
    end
end



# initializing state vector

Z = 2 * rand(system.vardim * T) .- 1


#
# testing objective derivatives
#


# definining objectives

quantum_obj = QuantumObjective(;
    system=system,
    cost_fn=cost_fn,
    T=T,
    Q=Q,
    eval_hessian
)

u_regularizer = QuadraticRegularizer(;
    indices=system.nstates .+ (1:system.ncontrols),
    vardim=system.vardim,
    times=1:T-1,
    R=R,
    eval_hessian=eval_hessian
)

mintime_obj = MinTimeObjective(;
    T=T,
    eval_hessian=true
)

u_smoothness_regularizer = QuadraticSmoothnessRegularizer(;
    indices=system.nstates .+ (1:system.ncontrols),
    vardim=system.vardim,
    times=1:T-1,
    R=Rᵤ,
    eval_hessian=true
)

@testset "objective and dynamics derivatives" begin

@testset "testing quantum objective + regularizer" begin

    obj = quantum_obj + u_regularizer



    # getting analytic gradient

    ∇L = obj.∇L(Z)



    # test gradient of objective with FiniteDiff

    # ∇L_finite_diff = FiniteDiff.finite_difference_gradient(obj.L, Z)

    # @test all(isapprox.(∇L, ∇L_finite_diff, atol=ATOL))


    # test gradient of objective with ForwardDiff

    ∇L_forward_diff = ForwardDiff.gradient(obj.L, Z)

    @test all(isapprox.(∇L, ∇L_forward_diff, atol=ATOL))


    # sparse objective Hessian data

    ∂²L = dense(
        obj.∂²L(Z),
        obj.∂²L_structure,
        (system.vardim * T, system.vardim * T)
    )



    # test hessian of objective with FiniteDiff

    # ∂²L_finite_diff = FiniteDiff.finite_difference_hessian(obj.L, Z)

    # show_diffs(∂²L, ∂²L_finite_diff)

    # @test all(isapprox.(∂²L, ∂²L_finite_diff, atol=ATOL))


    # test hessian of objective with ForwardDiff

    ∂²L_forward_diff = ForwardDiff.hessian(obj.L, Z)

    @test all(isapprox.(∂²L, ∂²L_forward_diff, atol=ATOL))


    #
    # testing dynamics derivatives
    #

    Δt = 0.01

    integrators = [:SecondOrderPade, :FourthOrderPade]

    for integrator in integrators

        # setting up dynamics struct

        dyns = QuantumDynamics(
            system,
            integrator,
            T,
            Δt;
            eval_hessian=eval_hessian
        )


        # dynamics Jacobian

        ∂F = dense(
            dyns.∂F(Z),
            dyns.∂F_structure,
            (system.nstates * (T - 1), system.vardim * T)
        )



        # test dynamics Jacobian vs finite diff

        # ∂F_finite_diff =
        #     FiniteDiff.finite_difference_jacobian(dyns.F, Z)

        # @test all(isapprox.(∂F, ∂F_finite_diff, atol=ATOL))


        # test dynamics Jacobian vs forward diff

        ∂F_forward_diff =
            ForwardDiff.jacobian(dyns.F, Z)

        @test all(isapprox.(∂F, ∂F_forward_diff, atol=ATOL))


        # Hessian of Lagrangian set up

        μ = randn(system.nstates * (T - 1))

        μ∂²F = dense(
            dyns.μ∂²F(μ, Z),
            dyns.μ∂²F_structure,
            (system.vardim * T, system.vardim * T)
        )

        HofL(Z) = dot(μ, dyns.F(Z))

        # test dynanamics Hessian of Lagrangian vs finite diff

        # HofL_finite_diff =
        #     FiniteDiff.finite_difference_hessian(HofL, Z)

        # @test all(isapprox.(μ∂²F, HofL_finite_diff, atol=ATOL))


        # test dynamics Hessian of Lagrangian vs forward diff

        HofL_forward_diff =
            ForwardDiff.hessian(HofL, Z)

        @test all(isapprox.(μ∂²F, HofL_forward_diff, atol=ATOL))

        show_diffs(μ∂²F, HofL_forward_diff)
    end

end


#
# test mintime objective derivatives
#

@testset "testing mintime objective + regularizer + smoothness regularizer " begin
    n_variables_mintime = system.vardim * T + T - 1

    Z_mintime = 2 * rand(n_variables_mintime) .- 1

    obj = mintime_obj + u_regularizer + u_smoothness_regularizer

    # getting analytic gradient

    ∇L = obj.∇L(Z_mintime)


    # test gradient of mintime objective with FiniteDiff

    # ∇L_finite_diff =
    #     FiniteDiff.finite_difference_gradient(
    #         obj.L,
    #         Z_mintime
    #     )

    # @test all(isapprox.(∇L, ∇L_finite_diff, atol=ATOL))

    # test gradient of mintime objective with ForwardDiff

    ∇L_forward_diff =
        ForwardDiff.gradient(obj.L, Z_mintime)

    @test all(isapprox.(∇L, ∇L_forward_diff, atol=ATOL))


    # sparse mintime objective Hessian data

    ∂²L = dense(
        obj.∂²L(Z_mintime),
        obj.∂²L_structure,
        (n_variables_mintime, n_variables_mintime)
    )

    # test hessian of mintime objective with FiniteDiff

    # ∂²L_finite_diff =
    #     FiniteDiff.finite_difference_hessian(
    #         obj.L,
    #         Z_mintime
    #     )

    # show_diffs(∂²L, ∂²L_finite_diff)

    # @test all(isapprox.(∂²L, ∂²L_finite_diff, atol=ATOL))


    # test hessian of mintime objective with ForwardDiff

    ∂²L_forward_diff =
        ForwardDiff.hessian(obj.L, Z_mintime)

    @test all(isapprox.(∂²L, ∂²L_forward_diff, atol=ATOL))

end

@testset "mintime dynamics" begin
    n_variables_mintime = system.vardim * T + T - 1
    Δt = rand(T - 1)
    Z̄ = [Z; Δt]

    μ = randn(system.nstates * (T - 1))

    Z_indices = 1:system.vardim * T
    Δt_indices = system.vardim * T .+ (1:T-1)

    Δt = 0.01

    integrators = [:SecondOrderPade, :FourthOrderPade]

    for integrator in integrators
        @info "integrator" integrator

        # setting up dynamics struct

        D = MinTimeQuantumDynamics(
            system,
            integrator,
            Z_indices,
            Δt_indices,
            T;
            eval_hessian=eval_hessian
        )

        ∂F = dense(
            D.∂F(Z̄),
            D.∂F_structure,
            (system.nstates * (T - 1), n_variables_mintime)
        )

        # test dynamics Jacobian vs forward diff

        ∂F_forward_diff =
            ForwardDiff.jacobian(D.F, Z̄)

        @test all(isapprox.(∂F, ∂F_forward_diff, atol=ATOL))

        show_diffs(∂F, ∂F_forward_diff)

        μ∂²F = dense(
            D.μ∂²F(μ, Z̄),
            D.μ∂²F_structure,
            (n_variables_mintime, n_variables_mintime)
        )

        HofL(Ẑ) = dot(μ, D.F(Ẑ))

        # test dynanamics Hessian of Lagrangian vs forward diff

        HofL_forward_diff =
            ForwardDiff.hessian(HofL, Z̄)

        @test all(isapprox.(μ∂²F, HofL_forward_diff, atol=ATOL))

        show_diffs(μ∂²F, HofL_forward_diff)
    end
end

@testset "slack variable objective" begin
    options = Options(
        max_iter=50,
        print_level=0,
    )

    slack_prob = QuantumControlProblem(
        system;
        L1_regularized_states=[1, system.isodim],
        options=options,
    )

    solve!(slack_prob)

    Z_slack_prob = get_variables(slack_prob)

    slack_obj = filter(slack_prob.objective_terms) do term
        term[:type] == :L1SlackRegularizer
    end[1] |> Objective

    ∇L = slack_obj.∇L(Z_slack_prob)

    # test gradient of slack variable objective with ForwardDiff

    ∇L_forward_diff =
        ForwardDiff.gradient(slack_obj.L, Z_slack_prob)

    @test all(isapprox.(∇L, ∇L_forward_diff; atol=ATOL))

    slack_con = filter!(slack_prob.constraints) do con
        con isa L1SlackConstraint
    end[1]

    # test slack constraint is satisfied on solved problem
    s1 = Z_slack_prob[slack_con.s1_indices]
    s2 = Z_slack_prob[slack_con.s2_indices]
    x = Z_slack_prob[slack_con.x_indices]

    @test all(isapprox.(s1 - s2, x; atol=ATOL))
end

end

"""
    testing saving and loading of problems
"""

@testset "testing saving and loading of problems" begin


    fixed_time_save_path = generate_file_path(
        "jld2",
        "test_save",
        pwd()
    )

    prob = QuantumControlProblem(system)

    save_problem(prob, fixed_time_save_path)

    min_time_save_path = generate_file_path(
        "jld2",
        "test_save",
        pwd()
    )

    min_time_prob = MinTimeQuantumControlProblem(system)

    save_problem(min_time_prob, min_time_save_path)



    @testset "fixed time problems" begin
        prob_loaded = load_problem(fixed_time_save_path)
        data_loaded = load_data(fixed_time_save_path)

        @test prob_loaded isa QuantumControlProblem
        @test data_loaded isa ProblemData

        @test data_loaded.type == :FixedTime
    end

    @testset "min time problems" begin
        min_time_prob_loaded = load_problem(min_time_save_path)
        min_time_data_loaded = load_data(min_time_save_path)


        @test min_time_prob_loaded isa MinTimeQuantumControlProblem
        @test min_time_data_loaded isa ProblemData
        @test min_time_data_loaded.type == :MinTime
    end

    @test min_time_save_path[1:end-6] * "0" == fixed_time_save_path[1:end-5]

    rm(fixed_time_save_path)
    rm(min_time_save_path)



end
