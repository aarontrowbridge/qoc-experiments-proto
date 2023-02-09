"""
    dynamics tests
"""

@testset "dynamics tests" begin

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

T = 10

Z = 2 * rand(system.vardim * T) .- 1

Δt = 0.01

integrators = [
    :SecondOrderPade,
    :FourthOrderPade
]

eval_hessian = true

@testset "testing $integrator dynamics" for integrator in integrators


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


@testset "mintime dynamics" begin
    n_variables_mintime = system.vardim * T + T - 1
    Δt = rand(T - 1)
    Z̄ = [Z; Δt]

    μ = randn(system.nstates * (T - 1))

    Z_indices = 1:system.vardim * T
    Δt_indices = system.vardim * T .+ (1:T-1)

    Δt = 0.01

    integrators = [:SecondOrderPade, :FourthOrderPade]

    @testset "testing $integrator mintime dynamics" for integrator in integrators
        # setting up dynamics struct

        D = QuantumDynamics(
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



end
