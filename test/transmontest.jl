using Pico
using Test

using ForwardDiff
using FiniteDiff
using SparseArrays
using LinearAlgebra


#
# setting up simple quantum system
#

ω = 2π * 4.96 #GHz
α = -2π * 0.143 #GHz
levels = 3

ψg = [1. + 0*im, 0 , 0]
ψe = [0, 1. + 0*im, 0]

ψ1 = [ψg, ψe]
ψf = [-im*ψe, -im*ψg]

H_drift = α/2 * quad(levels)
H_drive = [create(levels) + annihilate(levels),
1im * (create(levels) - annihilate(levels))]


system = QuantumSystem(
    H_drift,
    H_drive,
    ψ1 = ψ1,
    ψf = ψf,
    control_bounds = [2π * 19e-3, 2π * 19e-3]
)


T = 5
Q = 200.0
R = 2.0

eval_hessian = true

cost_fn = amplitude_cost


# absolulte tolerance for approximate tests

const ATOL = 1e-6


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


# initializing state vector

Z = 2 * rand(system.vardim * T) .- 1

z1 = 2*rand(system.vardim) .- 1

#
# testing objective derivatives
#


# setting up objective struct

obj = QuantumObjective(
    system,
    cost_fn,
    T,
    Q,
    R,
    eval_hessian
)

# getting analytic gradient

∇ = obj.∇L(Z)


# test gradient of objective with FiniteDiff

# @test all(
#     isapprox.(
#         ∇,
#         FiniteDiff.finite_difference_gradient(obj.L, Z),
#         atol=ATOL
#     )
# )


# # test gradient of objective with ForwardDiff

@test all(
    isapprox.(
        ∇,
        ForwardDiff.gradient(obj.L, Z),
        atol=ATOL
    )
)


# sparse objective Hessian data

H = dense(
    obj.∇²L(Z),
    obj.∇²L_structure,
    (system.vardim * T, system.vardim * T)
)

# display(H)


# # test hessian of objective with FiniteDiff

# # @test all(
# #     isapprox.(
# #         H,
# #         FiniteDiff.finite_difference_hessian(obj.L, Z),
# #         atol=ATOL
# #     )
# # )


# # test hessian of objective with ForwardDiff

@test all(
    isapprox.(
        H,
        ForwardDiff.hessian(obj.L, Z),
        atol=ATOL
    )
)


#
# testing dynamics derivatives
#

Δt = 0.1

#ntegrators = [:SecondOrderPade, :FourthOrderPade]
integrators = [:FourthOrderPade]

for integrator in integrators

    # setting up dynamics struct

    dyns = QuantumDynamics(
        system,
        integrator,
        T,
        Δt,
        eval_hessian
    )


    # dynamics Jacobian

    J = dense(
        dyns.∂F(Z),
        dyns.∂F_structure,
        (system.nstates * (T - 1), system.vardim * T)
    )



    println("Analytic dynamics Jacobian")
    display(J[1:16, 1:18])

    # test dynamics Jacobian vs finite diff

    # @test all(
    #     isapprox.(
    #         J,
    #         FiniteDiff.finite_difference_jacobian(dyns.F, Z),
    #         atol=ATOL
    #     )
    # )
    JFd = ForwardDiff.jacobian(dyns.F, Z)
    display(JFd[1:16, 1:18])

    # # test dynamics Jacobian vs forward diff

    @test all(
        isapprox.(
            J,
            ForwardDiff.jacobian(dyns.F, Z),
            atol=ATOL
        )
    )

    # res = isapprox.(
    #         J,
    #         JFd,
    #         atol=1e-2
    #     )

    # display(res)

    # @test all(
    #     isapprox.(
    #         J,
    #         JFd,
    #         atol = 1e-1
    #     )
    # )
    # # Hessian of Lagrangian set up

    μ = randn(system.nstates * (T - 1))

    m = dyns.μ∂²F(Z, μ)
    struc = dyns.μ∂²F_structure
    print("test")
    display(m)
    display(struc)
    μ∂²F = dense(
        m,
        struc,
        (system.vardim * T, system.vardim * T)
    )

    HofL(Z) = dot(μ, dyns.F(Z))

    # # for (H_analytic, H_numerical) in zip(
    # #     μ∂²F,
    # #     FiniteDiff.finite_difference_hessian(HofL, Z)
    # # )
    # #     if !isapprox(H_analytic, H_numerical, atol=ATOL)
    # #         println((H_analytic, H_numerical))
    # #     end
    # # end


    # # test dynamics Hessian of Lagrangian vs finite diff

    # # @test all(
    # #     isapprox.(
    # #         μ∂²F,
    # #         FiniteDiff.finite_difference_hessian(HofL, Z),
    # #         atol=ATOL
    # #     )
    # # )


    # # test dynamics Hessian of Lagrangian vs forward diff

    dH = ForwardDiff.hessian(HofL, Z)
    display(μ∂²F[1:18, 1:18])
    display(dH[1:18, 1:18])
    res2 = isapprox.(
            μ∂²F,
            dH,
            atol=1e-2
        )

    display(res2)
    @test all(
        isapprox.(
            μ∂²F,
            dH,
            atol=ATOL
        )
    )
end
