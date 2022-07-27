using QubitControl
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

ψ = [ψ0, ψ1, (ψ0 + im * ψ1) / √2, (ψ0 - ψ1) / √2]

system = SingleQubitSystem(
    H_drift,
    H_drive,
    gate, ψ
)

T = 2
Q = 200.0
R = 2.0
eval_hessian = true

loss_fn = amplitude_loss


# absolulte tolerance for approximate tests

const ATOL = 1e-2


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



#
# testing objective derivatives
#


# setting up objective struct

obj = SystemObjective(
    system,
    loss_fn,
    T,
    Q,
    R,
    eval_hessian
)


# initializing state vector

Z = 2 * rand(system.vardim * T) .- 1

∇ = obj.∇L(Z)


# test gradient of objective with FiniteDiff

@test isapprox(
    ∇,
    FiniteDiff.finite_difference_gradient(obj.L, Z),
    atol=ATOL
)


# test gradient of objective with ForwardDiff

@test isapprox(
    ∇,
    ForwardDiff.gradient(obj.L, Z),
    atol=ATOL
)


# sparse objective Hessian data

H = dense(
    obj.∇²L(Z),
    obj.∇²L_structure,
    (system.vardim * T, system.vardim * T)
)


# test hessian of objective with FiniteDiff

@test all(
    isapprox.(
        FiniteDiff.finite_difference_hessian(obj.L, Z),
        H,
        atol=ATOL
    )
)


# test hessian of objective with ForwardDiff

@test all(
    isapprox.(
        ForwardDiff.hessian(obj.L, Z),
        H,
        atol=ATOL
    )
)


#
# testing dynamics derivatives
#

Δt = 0.01

integrators = [:SecondOrderPade, :FourthOrderPade]

for integrator in integrators
    dyns = SystemDynamics(
        system,
        integrator,
        T,
        Δt,
        eval_hessian
    )


    # dynamics Jacobian

    J = dense(
        dyns.∇F(Z),
        dyns.∇F_structure,
        (system.nstates * (T - 1), system.vardim * T)
    )



    # test dynamics Jacobian vs finite diff

    @test all(
        isapprox.(
            FiniteDiff.finite_difference_jacobian(dyns.F, Z),
            J,
            atol=ATOL
        )
    )


    # test dynamics Jacobian vs forward diff

    @test all(
        isapprox.(
            ForwardDiff.jacobian(dyns.F, Z),
            J,
            atol=ATOL
        )
    )


    # Hessian of Lagrangian set up

    μ = randn(system.nstates * (T - 1))

    μ∇²F = dense(
        dyns.μ∇²F(Z, μ),
        dyns.μ∇²F_structure,
        (system.vardim * T, system.vardim * T)
    )

    HofL(Z) = dot(μ, dyns.F(Z))

    for (H_analytic, H_numerical) in zip(
        μ∇²F,
        FiniteDiff.finite_difference_hessian(HofL, Z)
    )
        if !isapprox(H_analytic, H_numerical, atol=ATOL)
            println((H_analytic, H_numerical))
        end
    end


    # test dynamics Hessian of Lagrangian vs finite diff

    @test all(
        isapprox.(
            μ∇²F,
            FiniteDiff.finite_difference_hessian(HofL, Z),
            atol=ATOL
        )
    )


    # test dynamics Hessian of Lagrangian vs forward diff

    @test all(
        isapprox.(
            μ∇²F,
            ForwardDiff.hessian(HofL, Z),
            atol=ATOL
        )
    )
end
