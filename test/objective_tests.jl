"""
    tests of objective functions


"""

@testset "objective tests" begin

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

a_bounds = [1.0, 0.5]

system = QuantumSystem(
    H_drift,
    H_drive,
    ψ,
    ψf;
    a_bounds=a_bounds,
)


T = 5

@assert T > 4 "mintime objective Hessian is set up for T > 4"

Q = 200.0
R = 1.0
Rᵤ = 1.0
Rₛ = 1.0



eval_hessian = true

cost_fn = :infidelity_cost


Z = 2 * rand(system.vardim * T) .- 1


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

u_smoothness_regularizer = QuadraticSmoothnessRegularizer(;
    indices=system.nstates .+ (1:system.ncontrols),
    vardim=system.vardim,
    times=1:T-1,
    R=Rᵤ,
    eval_hessian=true
)

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




end


#
# test mintime objective derivatives
#

@testset "testing mintime objective + regularizer + smoothness regularizer " begin
    n_variables_mintime = system.vardim * T + T

    Z_mintime = 2 * rand(n_variables_mintime) .- 1

    mintime_obj = MinTimeObjective(;
        Δt_indices=system.vardim * T .+ (1:T),
        T=T
    )

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

@testset "slack variable objective" begin
    options = Options(
        max_iter=50,
        print_level=0
    )

    slack_prob = QuantumControlProblem(
        system;
        L1_regularized_states=[1, system.isodim],
        options=options,
    )

    solve!(slack_prob)

    Z_slack_prob = get_variables(slack_prob)

    slack_obj = filter(slack_prob.params[:objective_terms]) do term
        term[:type] == :L1SlackRegularizer
    end[1] |> Objective

    ∇L = slack_obj.∇L(Z_slack_prob)

    # test gradient of slack variable objective with ForwardDiff

    ∇L_forward_diff =
        ForwardDiff.gradient(slack_obj.L, Z_slack_prob)

    @test all(isapprox.(∇L, ∇L_forward_diff; atol=ATOL))

    # test slack constraint is satisfied on solved problem
    s1 = Z_slack_prob[slack_prob.params[:s1_indices]]
    s2 = Z_slack_prob[slack_prob.params[:s2_indices]]
    x = Z_slack_prob[slack_prob.params[:x_indices]]

    @test all(isapprox.(s1 - s2, x; atol=ATOL))
end




end
