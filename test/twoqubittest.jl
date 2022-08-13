using QubitControl
using Test

using ForwardDiff
using FiniteDiff
using SparseArrays
using LinearAlgebra


#
# setting up simple quantum system
#

ω1 =  2π * 3.5 #GHz
ω2 = 2π * 3.9 #GHz
J = 0.1 * 2π

ψ1 = [[0, 0, 0, 1 + 0im],
      [1.0  + 0im, 0, 0, 0],
      [0, 1. + 0im, 0, 0],
      [0, 0, 1 + 0im, 0]
      ]
ψf = apply.(:sqrtiSWAP, ψ1)


    # H_drift = -(ω1/2 + gcouple)*kron(GATES[:Z], I(2)) - 
    #            (ω2/2 + gcouple)*kron(I(2), GATES[:Z]) +
    #            gcouple*kron(GATES[:Z], GATES[:Z])
    # H_drift = ω1*kron(number(2), I(2)) + ω2*kron(I(2), number(2)) + 
    #            J*(kron(GATES[:Z], GATES[:Z]))
    # H_drift = J*kron(GATES[:Z], GATES[:Z])
H_drift = zeros(4,4) #ω1/2 * kron(GATES[:Z], I(2)) + ω2/2 * kron(I(2), GATES[:Z]

H_drive = [kron(create(2), annihilate(2)) + kron(annihilate(2), create(2)),
           1im*(kron(create(2), annihilate(2)) - kron(annihilate(2), create(2)))]


system = QuantumSystem(
    H_drift,
    H_drive,
    ψ1 = ψ1,
    ψf = ψf,
    control_bounds = ones(length(H_drive))
)



T = 5

@assert T > 4 "mintime objective Hessian is set up for T > 4"

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


# show differences between arrays

function show_diffs(A, B)
    for (i, (a, b)) in enumerate(zip(A, B))
        inds = Tuple(CartesianIndices(A)[i])
        if !isapprox(a, b, atol=ATOL) && inds[1] ≤ inds[2]
            @info "values" (a, b)
            @info "indices" inds
        end
    end
end



# initializing state vector

Z = 2 * rand(system.vardim * T) .- 1


#
# testing objective derivatives
#


# setting up objective struct

obj = SystemObjective(
    system,
    cost_fn,
    T,
    Q,
    R,
    eval_hessian
)

# getting analytic gradient

∇L = obj.∇L(Z)



# test gradient of objective with FiniteDiff

# ∇L_finite_diff = FiniteDiff.finite_difference_gradient(obj.L, Z)

# @test all(isapprox.(∇L, ∇L_finite_diff, atol=ATOL))


# test gradient of objective with ForwardDiff

∇L_forward_diff = ForwardDiff.gradient(obj.L, Z)

@test all(isapprox.(∇L, ∇L_forward_diff, atol=ATOL))


# sparse objective Hessian data

∇²L = dense(
    obj.∇²L(Z),
    obj.∇²L_structure,
    (system.vardim * T, system.vardim * T)
)



# test hessian of objective with FiniteDiff

# ∇²L_finite_diff = FiniteDiff.finite_difference_hessian(obj.L, Z)

# show_diffs(∇²L, ∇²L_finite_diff)

# @test all(isapprox.(∇²L, ∇²L_finite_diff, atol=ATOL))


# test hessian of objective with ForwardDiff

∇²L_forward_diff = ForwardDiff.hessian(obj.L, Z)

@test all(isapprox.(∇²L, ∇²L_forward_diff, atol=ATOL))


#
# testing dynamics derivatives
#

Δt = 0.01

integrators = [:SecondOrderPade, :FourthOrderPade]

for integrator in integrators

    # setting up dynamics struct

    dyns = SystemDynamics(
        system,
        integrator,
        T,
        Δt,
        eval_hessian
    )


    # dynamics Jacobian

    ∇F = dense(
        dyns.∇F(Z),
        dyns.∇F_structure,
        (system.nstates * (T - 1), system.vardim * T)
    )



    # test dynamics Jacobian vs finite diff

    # ∇F_finite_diff =
    #     FiniteDiff.finite_difference_jacobian(dyns.F, Z)

    # @test all(isapprox.(∇F, ∇F_finite_diff, atol=ATOL))


    # test dynamics Jacobian vs forward diff

    ∇F_forward_diff =
        ForwardDiff.jacobian(dyns.F, Z)

    @test all(isapprox.(∇F, ∇F_forward_diff, atol=ATOL))


    # Hessian of Lagrangian set up

    μ = randn(system.nstates * (T - 1))

    μ∇²F = dense(
        dyns.μ∇²F(Z, μ),
        dyns.μ∇²F_structure,
        (system.vardim * T, system.vardim * T)
    )

    HofL(Z) = dot(μ, dyns.F(Z))

    # test dynanamics Hessian of Lagrangian vs finite diff

    # HofL_finite_diff =
    #     FiniteDiff.finite_difference_hessian(HofL, Z)

    # @test all(isapprox.(μ∇²F, HofL_finite_diff, atol=ATOL))


    # test dynamics Hessian of Lagrangian vs forward diff

    HofL_forward_diff =
        ForwardDiff.hessian(HofL, Z)

    @test all(isapprox.(μ∇²F, HofL_forward_diff, atol=ATOL))
end


#
# test mintime objective derivatives
#

Z_mintime = 2 * rand(system.vardim * T + T - 1) .- 1

Rᵤ = 1e-3
Rₛ = 1e-3

mintime_obj = MinTimeObjective(
    system,
    T,
    Rᵤ,
    Rₛ,
    true
)

# getting analytic gradient

∇L = mintime_obj.∇L(Z_mintime)


# test gradient of mintime objective with FiniteDiff

# ∇L_finite_diff =
#     FiniteDiff.finite_difference_gradient(
#         mintime_obj.L,
#         Z_mintime
#     )

# @test all(isapprox.(∇L, ∇L_finite_diff, atol=ATOL))

# test gradient of mintime objective with ForwardDiff

∇L_forward_diff =
    ForwardDiff.gradient(mintime_obj.L, Z_mintime)

@test all(isapprox.(∇L, ∇L_forward_diff, atol=ATOL))


# sparse mintime objective Hessian data

∇²L = dense(
    mintime_obj.∇²L(Z_mintime),
    mintime_obj.∇²L_structure,
    (system.vardim * T + T - 1, system.vardim * T + T - 1)
)

# test hessian of mintime objective with FiniteDiff

# ∇²L_finite_diff =
#     FiniteDiff.finite_difference_hessian(
#         mintime_obj.L,
#         Z_mintime
#     )

# show_diffs(∇²L, ∇²L_finite_diff)

# @test all(isapprox.(∇²L, ∇²L_finite_diff, atol=ATOL))


# test hessian of mintime objective with ForwardDiff

∇²L_forward_diff =
    ForwardDiff.hessian(mintime_obj.L, Z_mintime)

@test all(isapprox.(∇²L, ∇²L_forward_diff, atol=ATOL))
