module InterativeLearningControl

export QuantumILCProblem

using ..Problems
using ..Trajectories
using ..QuantumSystems
using ..Dynamics
using ..Integrators

using LinearAlgebra
using SparseArrays
using OSQP

struct MeasurementData
    ys::Vector{Vector{Float64}}
    times::AbstractVector{Int}
    ydim::Int
end

function measure(
    Z::Trajectory,
    g::Function,
    times::AbstractVector{Int},
    ydim::Int,
)
    @assert size(g(Z.states[1]), 1) == ydim
    ys = Vector{Vector{Float64}}(undef, length(times))
    for (i, t) in enumerate(times)
        ys[i] = g(Z.states[t])
    end
    return MeasurementData(ys, times, ydim)
end

abstract type ILCProblem end

struct QuantumILCProblem <: ILCProblem
    system::QuantumSystem
    Ẑ::Trajectory
    g::Function
    Ŷ::MeasurementData
    Ȳ::MeasurementData
    H::SparseMatrixCSC{Float64, Int}
    A::SparseMatrixCSC{Float64, Int}

end


function QuantumILCProblem(
    sys::QuantumSystem,
    Ẑ::Trajectory,
    g::Function,
    Ȳ::MeasurementData;
    Q=1.0,
    R=1.0,
    integrator=:FourthOrderPade
)
    Ŷ = measure(Ẑ, g, Ȳ.times, Ȳ.ydim)
    H = build_hessian(Q, R, sys.vardim, sys.ncontrols, Ẑ.T)
    A = build_constraint_matrix(sys, Ẑ, g, Ŷ, Ȳ, integrator)
    return QuantumILCProblem(sys, Ẑ, g, Ŷ, Ȳ, H, A)
end

function build_hessian(
    Q::Float64,
    R::Float64,
    xdim::Int,
    udim::Int,
    T::Int
)
    Hₜ = spdiagm(0 => [Q * ones(xdim); R * ones(udim)])
    H = kron(sparse(I(T)), Hₜ)
    return H
end

function build_constraint_matrix(
    sys::QuantumSystem,
    X̂::Trajectory,
    g::Function,
    Ŷ::MeasurementData,
    Ȳ::MeasurementData,
    integrator::Symbol
)
    ∇f̂s = build_dynamics_jacobians(sys, X̂, integrator)
    ∇ḡs = build_measurement_jacobians(sys, g, Ȳ)

    As = []

    A₁ = sparse_vcat(
        ∇f̂s[1],
        sparse_hcat(
            ∇ḡs[1],
            spzeros(
                Ŷ.ydim,
                sys.n_aug_states + sys.ncontrols
            )
        ),
        sparse_hcat(
            spzeros(
                sys.n_aug_states + sys.ncontrols,
                sys.n_wfn_states
            ),
            sparse(I(sys.n_aug_states + sys.ncontrols))
        )
    )

    push!(As, A₁)

    for t = 2:X̂.T-1
        Aₜ = sparse_vcat(
            ∇f̂s[t],
            sparse_hcat(
                ∇ḡs[t],
                spzeros(
                    Ŷ.ydim,
                    sys.n_aug_states + sys.ncontrols
                )
            ),
        )
        push!(As, Aₜ)
    end

    A_T = sparse_vcat(
        sparse_hcat(
            ∇ḡs[X̂.T],
            spzeros(
                Ŷ.ydim,
                sys.n_aug_states + sys.ncontrols
            )
        ),
        sparse_hcat(
            spzeros(
                sys.n_aug_states + sys.ncontrols,
                sys.n_wfn_states
            ),
            sparse(I(sys.n_aug_states + sys.ncontrols))
        )
    )

    push!(As, A_T)

    A = foldr(blockdiag, As)

    return A
end


end
