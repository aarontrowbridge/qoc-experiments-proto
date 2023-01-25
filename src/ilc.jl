module IterativeLearningControl

export ILCProblem
export ILCProblemNew
export solve!

using ..Trajectories
using ..QuantumSystems
using ..Integrators
using ..Utils
using ..ProblemSolvers
using ..QuantumLogic
using ..Dynamics
using ..ILCExperiments
using ..ILCQuadraticProblems
using ..ILCTrajectories

using LinearAlgebra
using SparseArrays
using Statistics

mutable struct ILCProblemNew
    Ẑgoal::Traj
    Ẑ::Traj
    QP::QuadraticProblem
    Ygoal::MeasurementData
    Ȳs::Vector{MeasurementData}
    As::Vector{Matrix{Float64}}
    experiment::AbstractExperiment
    settings::Dict{Symbol, Any}
    bt_dict::Dict{Int, Vector}

    function ILCProblemNew(
        Ẑgoal::Traj,
        dynamics::Function,
        experiment::QuantumExperiment;
        Qy=1.0,
        Qyf=100.0,
        R=(a=1.0, dda=1.0),
        correction_term=true,
        dynamic_measurements=true,
        verbose=true,
        max_iter=100,
        max_backtrack_iter=10,
        tol=1e-6,
        α=0.5,
        β=0.5,
        norm_p=Inf,
        static_QP=false,
        QP_max_iter=100_000,
        QP_verbose=false,
        QP_linear_solver="mkl pardiso",
        QP_settings=Dict(),
        QP_tol=1e-9,
        mle=true,
        Σ=nothing,
    )

        @assert R isa NamedTuple "R must be a NamedTuple specifying components to regularize -- e.g. R=(a=1.0, dda=1.0)"

        ILC_settings::Dict{Symbol, Any} = Dict(
            :max_iter => max_iter,
            :max_backtrack_iter => max_backtrack_iter,
            :α => α,
            :β => β,
            :tol => tol,
            :norm_p => norm_p,
            :verbose => verbose,
        )

        QP_kw_settings::Dict{Symbol, Any} = Dict(
            :max_iter => QP_max_iter,
            :verbose => QP_verbose,
            :eps_abs => QP_tol,
            :eps_rel => QP_tol,
            :eps_prim_inf => QP_tol,
            :eps_dual_inf => QP_tol,
            :linsys_solver => QP_linear_solver,
        )

        QP_settings::Dict{Symbol, Any} =
            merge(QP_kw_settings, QP_settings)

        @assert length(dynamics(Ẑgoal[1].data, Ẑgoal[2].data, Ẑgoal)) ==
            Ẑgoal.dims.states

        f = zₜzₜ₊₁::AbstractVector{<:Real} -> begin
            zₜ = zₜzₜ₊₁[slice(1, Ẑgoal.dim)]
            zₜ₊₁ = zₜzₜ₊₁[slice(2, Ẑgoal.dim)]
            return dynamics(zₜ, zₜ₊₁, Ẑgoal)
        end

        if static_QP
            QP = StaticQuadraticProblemNew(
                Ẑgoal,
                f,
                experiment.g,
                experiment.τs,
                Qy, Qyf, R,
                correction_term,
                dynamic_measurements,
                QP_settings;
                mle=mle
            )
        else
            QP = DynamicQuadraticProblemNew(
                Ẑgoal,
                f,
                experiment.g,
                Qy, Qyf, R,
                correction_term,
                QP_settings;
                mle=mle
            )
        end

        Ygoal = measure(
            Ẑgoal.ψ̃,
            experiment.g,
            experiment.τs,
            experiment.ydim
        )



        # display(Ygoal.ys[end])
        # println()
        # display(Ẑgoal.states[end])

        return new(
            Ẑgoal,
            Ẑgoal,
            QP,
            Ygoal,
            MeasurementData[],
            Matrix{Float64}[],
            experiment,
            ILC_settings,
            Dict{Int, Vector}()
        )
    end
end

function ProblemSolvers.solve!(
    prob::ILCProblemNew;
    σ=1.0,
    nominal_correction=true,
    nominal_rollout_integrator=Integrators.fourth_order_pade,
)
    # get the initial controls and add to the stored list of controls
    A = prob.Ẑ.a
    push!(prob.As, A)

    # get the initial measurement and add to the stored list of measurements
    Ȳ = prob.experiment(A, times(prob.Ẑ); σ=σ)
    push!(prob.Ȳs, Ȳ)

    # compute the initial error
    ΔY = Ȳ - prob.Ygoal

    println()

    # initialize the tracked list of normed measurement errors
    ΔyT_norms = [norm(ΔY.ys[end], prob.settings[:norm_p])]

    # initialize the iteration counter
    k = 1

    while true

        # check if we've reached the maximum number of ILC iterations
        if k > prob.settings[:max_iter]
            @info "max iterations reached" max_iter = prob.settings[:max_iter]
            return
        end

        # printed ILC iteratation information
        if prob.settings[:verbose]
            println()
            printstyled("iter    = ", k; color=:magenta)
            println()
            # printstyled("⟨|ΔY|⟩  = ", mean([norm(y, prob.settings[:norm_p]) for y in ΔY.ys]); color=:magenta)
            # println()
            printstyled("|ΔY(T)| = ", norm(ΔY.ys[end], prob.settings[:norm_p]); color=:magenta)
            println()
            println()
        end

        # get step direction from QP
        ΔZ = prob.QP(ΔY, prob.Ẑ)

        # mutliply by pre-scaling factor β
        ΔZ *= prob.settings[:β]

        # update nominal trajectory
        println("norm(A) = ", norm(prob.Ẑ.a))
        prob.Ẑ += ΔZ
        println("norm(A) = ", norm(prob.Ẑ.a))

        # get the updated controls
        A = prob.Ẑ.a

        # run an experiment with the updated controls
        Ȳ = prob.experiment(A, times(prob.Ẑ); σ=σ)

        # compute the difference between the updated measurements and the goal
        ΔY_next = Ȳ - prob.Ygoal # type: MeasurementData

        # get the vector valued final measurement difference
        ΔyT_next = ΔY_next.ys[end]

        # line search phase of ILC
        # TODO: make this modular
        if norm(ΔyT_next, prob.settings[:norm_p]) >= minimum(ΔyT_norms)

            printstyled("   backtracking"; color=:magenta)
            println()
            println()

            # initialize the backtracking iteration counter
            i = 1

            # initialize the list of final measurements during backtracking
            backtrack_yTs = []

            # initialize the list of measurement errors during backtracking
            iter_ΔyT_norms = []

            # backtracking loop
            while norm(ΔyT_next, prob.settings[:norm_p]) >= minimum(ΔyT_norms)

                # check if we've reached the maximum number of backtracking iterations
                if i > prob.settings[:max_backtrack_iter]
                    println()
                    printstyled("   max backtracking iterations reached"; color=:magenta)
                    println()
                    println()
                    ΔY = ΔY_next
                    return
                end

                # reset to previous state
                prob.Ẑ -= ΔZ
                println("norm(A) = ", norm(prob.Ẑ.a))

                # decrease step size by a factor of α
                ΔZ *= prob.settings[:α]

                # take step
                prob.Ẑ += ΔZ
                println("norm(A) = ", norm(prob.Ẑ.a))

                # get pulse controls
                A = prob.Ẑ.a

                # get the final measurement of the experiment with the updated controls
                yT = prob.experiment(A, times(prob.Ẑ); backtracking=true, σ=σ).ys[end]

                display(minimum(abs.(yT)))

                # store the final measurement
                push!(backtrack_yTs, yT)

                # compute the difference between the updated measurement and the goal
                ΔyT_next = yT - prob.Ygoal.ys[end]

                # store the norm of the final measurement difference
                push!(iter_ΔyT_norms, norm(ΔyT_next, prob.settings[:norm_p]))

                # printed backtracking iteration information
                println()
                printstyled("       bt_iter     = ", i; color=:cyan)
                println()
                printstyled("       min |ΔY(T)| = ", minimum(ΔyT_norms); color=:cyan)
                println()
                printstyled("       |ΔY(T)|     = ", norm(ΔyT_next, prob.settings[:norm_p]); color=:cyan)
                println()
                println()

                # increment backtracking iteration counter
                i += 1
            end

            # update the list of normed measurement errors
            # with the minimum of the backtracking iteration errors
            push!(ΔyT_norms, minimum(iter_ΔyT_norms))

            # store the backtracking per iteration measurement data
            prob.bt_dict[k] = backtrack_yTs

            # remeasure with new controls to get full measurement
            Ȳ = prob.experiment(A, times(prob.Ẑ); σ=σ)

            # compute new error
            ΔY = Ȳ - prob.Ygoal

            # push remeasured norm(ΔY) to tracked errors
            push!(ΔyT_norms, norm(ΔY.ys[end], prob.settings[:norm_p]))
        else
            # if not backtracking, just update the error
            ΔY = ΔY_next

            # push normed final error to tracked errors
            push!(ΔyT_norms, norm(ΔY.ys[end], prob.settings[:norm_p]))
        end

        # store the updated measurement data
        push!(prob.Ȳs, Ȳ)

        # store the updated controls
        push!(prob.As, A)

        if nominal_correction
            # get corrected nominal trajectory
            Ψ̃_corrected = nominal_rollout(A, times(prob.Ẑ), prob.experiment; integrator=nominal_rollout_integrator)

            # update nominal trajectory
            update!(prob.Ẑ, :ψ̃, Ψ̃_corrected)
        end

        # increment ILC iteration counter
        k += 1
    end
end



function random_cov_matrix(n::Int; σ=1.0)
    σs = σ * rand(n)
    sort!(σs, rev=true)
    Q = qr(randn(n, n)).Q
    return Q * Diagonal(σs) * Q'
end


mutable struct ILCProblem
    Ẑgoal::Trajectory
    Ẑ::Trajectory
    QP::QuadraticProblem
    Ygoal::MeasurementData
    Ȳs::Vector{MeasurementData}
    Us::Vector{Matrix{Float64}}
    experiment::AbstractExperiment
    settings::Dict{Symbol, Any}
    d2u::Bool
    bt_dict::Dict{Int, Vector}

    function ILCProblem(
        sys::QuantumSystem,
        Ẑgoal::Trajectory,
        experiment::HardwareExperiment;
        integrator=:FourthOrderPade,
        Q=0.0,
        Qy=1.0,
        Qf=100.0,
        R=1.0,
        # identity matrix of Float64
        Σ=Diagonal(fill(1.0, experiment.ydim)),
        u_bounds=sys.a_bounds,
        correction_term=true,
        verbose=true,
        max_iter=100,
        max_backtrack_iter=10,
        tol=1e-6,
        α=0.5,
        β=0.5,
        norm_p=Inf,
        static_QP=false,
        QP_max_iter=1e5,
        QP_verbose=false,
        QP_tol=1e-9,
        QP_linear_solver="mkl pardiso",
        QP_settings=Dict(),
        d2u=false,
        d2u_bounds=fill(1e-5, length(u_bounds)),
        use_system_goal=false,
    )
        @assert length(u_bounds) == sys.ncontrols

        ILC_settings::Dict{Symbol, Any} = Dict(
            :max_iter => max_iter,
            :max_backtrack_iter => max_backtrack_iter,
            :α => α,
            :β => β,
            :tol => tol,
            :norm_p => norm_p,
            :verbose => verbose,
        )

        QP_kw_settings::Dict{Symbol, Any} = Dict(
            :max_iter => QP_max_iter,
            :verbose => QP_verbose,
            :eps_abs => QP_tol,
            :eps_rel => QP_tol,
            :eps_prim_inf => QP_tol,
            :eps_dual_inf => QP_tol,
            :linsys_solver => QP_linear_solver,
        )

        QP_settings::Dict{Symbol, Any} =
            merge(QP_kw_settings, QP_settings)

        dims = (
            x=size(Ẑgoal.states[1], 1),
            u=size(Ẑgoal.actions[1], 1),
            z=size(Ẑgoal.states[1], 1) + size(Ẑgoal.actions[1], 1),
            y=experiment.ydim,
            T=Ẑgoal.T,
            M=length(experiment.τs)
        )

        if d2u
            f = zₜzₜ₊₁ -> begin
                xₜ = zₜzₜ₊₁[1:dims.x]
                uₜ = zₜzₜ₊₁[dims.x .+ (1:dims.u)]
                xₜ₊₁ = zₜzₜ₊₁[dims.z .+ (1:dims.x)]
                return Dynamics.dynamics_sep(xₜ₊₁, xₜ, uₜ, Ẑgoal.Δt, eval(integrator)(sys), sys)
            end
        else
            dynamics = eval(integrator)(sys)

            f = zₜzₜ₊₁ -> begin
                xₜ = zₜzₜ₊₁[1:dims.x]
                uₜ = zₜzₜ₊₁[dims.x .+ (1:dims.u)]
                xₜ₊₁ = zₜzₜ₊₁[dims.z .+ (1:dims.x)]
                return dynamics(xₜ₊₁, xₜ, uₜ, Ẑgoal.Δt)
            end
        end

        if static_QP
            QP = StaticQuadraticProblem(
                Ẑgoal,
                f, experiment.g_analytic,
                Q, Qy, Qf, R, Σ,
                u_bounds,
                correction_term,
                QP_settings,
                dims
            )
        else
            QP = DynamicQuadraticProblem(
                f, experiment.g_analytic,
                Q, Qy, Qf, R, Σ,
                u_bounds,
                correction_term,
                QP_settings,
                dims
            )
        end

        Ygoal = measure(
            Ẑgoal,
            experiment.g_analytic,
            experiment.τs,
            experiment.ydim
        )

        if use_system_goal
            Ygoal.ys[end] = experiment.g_analytic(sys.ψ̃goal[slice(1, sys.isodim)])
        end

        # display(Ygoal.ys[end])
        # println()
        # display(Ẑgoal.states[end])

        return new(
            Ẑgoal,
            Ẑgoal,
            QP,
            Ygoal,
            MeasurementData[],
            Matrix{Float64}[],
            experiment,
            ILC_settings,
            Dict{Int, Vector}()
        )
    end

    function ILCProblem(
        sys::QuantumSystem,
        Ẑgoal::Trajectory,
        experiment::QuantumExperiment;
        integrator=:FourthOrderPade,
        Q=0.0,
        Qy=1.0,
        Qf=100.0,
        R=1.0,
        # identity matrix of Float64
        Σ=Diagonal(fill(1.0, experiment.ydim)),
        u_bounds=sys.a_bounds,
        correction_term=true,
        verbose=true,
        max_iter=100,
        max_backtrack_iter=10,
        tol=1e-6,
        α=0.5,
        β=0.5,
        norm_p=Inf,
        static_QP=false,
        QP_max_iter=100_000,
        QP_verbose=false,
        QP_tol=1e-9,
        QP_linear_solver="mkl pardiso",
        QP_settings=Dict(),
        d2u=false,
        d2u_bounds=fill(1e-5, length(u_bounds)),
        use_system_goal=false,
    )
        @assert length(u_bounds) == sys.ncontrols

        ILC_settings::Dict{Symbol, Any} = Dict(
            :max_iter => max_iter,
            :max_backtrack_iter => max_backtrack_iter,
            :α => α,
            :β => β,
            :tol => tol,
            :norm_p => norm_p,
            :verbose => verbose,
        )

        QP_kw_settings::Dict{Symbol, Any} = Dict(
            :max_iter => QP_max_iter,
            :verbose => QP_verbose,
            :eps_abs => QP_tol,
            :eps_rel => QP_tol,
            :eps_prim_inf => QP_tol,
            :eps_dual_inf => QP_tol,
            :linsys_solver => QP_linear_solver,
        )

        QP_settings::Dict{Symbol, Any} =
            merge(QP_kw_settings, QP_settings)

        dims = (
            x=size(Ẑgoal.states[1], 1),
            u=size(Ẑgoal.actions[1], 1),
            z=size(Ẑgoal.states[1], 1) + size(Ẑgoal.actions[1], 1),
            y=experiment.ydim,
            T=Ẑgoal.T,
            M=length(experiment.τs)
        )

        if d2u
            f = zₜzₜ₊₁ -> begin
                xₜ = zₜzₜ₊₁[1:dims.x]
                uₜ = zₜzₜ₊₁[dims.x .+ (1:dims.u)]
                xₜ₊₁ = zₜzₜ₊₁[dims.z .+ (1:dims.x)]
                return Dynamics.dynamics_sep(xₜ₊₁, xₜ, uₜ, Ẑgoal.Δt, eval(integrator)(sys), sys)
            end
        else
            dynamics = eval(integrator)(sys)

            f = zₜzₜ₊₁ -> begin
                xₜ = zₜzₜ₊₁[1:dims.x]
                uₜ = zₜzₜ₊₁[dims.x .+ (1:dims.u)]
                xₜ₊₁ = zₜzₜ₊₁[dims.z .+ (1:dims.x)]
                return dynamics(xₜ₊₁, xₜ, uₜ, Ẑgoal.Δt)
            end
        end

        if static_QP
            QP = StaticQuadraticProblem(
                Ẑgoal,
                f,
                experiment.g,
                experiment.τs,
                Q, Qy, Qf, R, Σ,
                u_bounds,
                correction_term,
                QP_settings,
                dims,
                d2u = d2u,
                d2u_bounds = d2u_bounds
            )
        else
            QP = DynamicQuadraticProblem(
                f,
                experiment.g,
                Q, Qy, Qf, R, Σ,
                u_bounds,
                correction_term,
                QP_settings,
                dims,
                d2u=d2u,
                d2u_bounds=d2u_bounds
            )
        end

        Ygoal = measure(
            Ẑgoal,
            experiment.g,
            experiment.τs,
            experiment.ydim
        )

        if use_system_goal
            Ygoal.ys[end] = experiment.g(sys.ψ̃goal[slice(1, sys.isodim)])
        end

        # display(Ygoal.ys[end])
        # println()
        # display(Ẑgoal.states[end])

        return new(
            Ẑgoal,
            Ẑgoal,
            QP,
            Ygoal,
            MeasurementData[],
            Matrix{Float64}[],
            experiment,
            ILC_settings,
            d2u,
            Dict{Int, Vector}()
        )
    end
end

function fidelity(ψ̃, ψ̃goal)
    ψ = iso_to_ket(ψ̃)
    ψgoal = iso_to_ket(ψ̃goal)
    # println("norm(ψ)     = $(norm(ψ))")
    # println("norm(ψgoal) = $(norm(ψgoal))")
    return abs2(ψ' * ψgoal)
end


function ProblemSolvers.solve!(prob::ILCProblem)
    U = prob.Ẑ.actions
    udim = length(U[1])
    A = [state[end - 2*udim + 1 : end - udim] for state in prob.Ẑ.states]
    U = hcat(U...)
    A = hcat(A...)
    Ȳ = prob.experiment(prob.d2u ? A : U, prob.Ẑ.times)
    push!(prob.Ȳs, Ȳ)
    push!(prob.Us, prob.d2u ? A : U)
    ΔY = Ȳ - prob.Ygoal
    println()
    printstyled()
    ΔyT_norms = [norm(ΔY.ys[end], prob.settings[:norm_p])]
    k = 1
    while true
        if k > prob.settings[:max_iter]
            @info "max iterations reached" max_iter = prob.settings[:max_iter]
            return
        end
        if prob.settings[:verbose]
            println()
            printstyled("iter    = ", k; color=:magenta)
            println()
            # printstyled("⟨|ΔY|⟩  = ", mean([norm(y, prob.settings[:norm_p]) for y in ΔY.ys]); color=:magenta)
            # println()
            printstyled("|ΔY(T)| = ", norm(ΔY.ys[end], prob.settings[:norm_p]); color=:magenta)
            println()
            println()
        end
        ΔZ = prob.settings[:β] * prob.QP(prob.Ẑ, ΔY)
        prob.Ẑ = prob.Ẑ + ΔZ

        us = prob.Ẑ.actions
        as = [state[end - 2*udim + 1 : end - udim] for state in prob.Ẑ.states]
        U = hcat(us...)
        A = hcat(as...)
        Ȳ = prob.experiment(prob.d2u ? A : U, prob.Ẑ.times)
        ΔYnext = Ȳ - prob.Ygoal
        ΔyTnext = ΔYnext.ys[end]

        # backtracking
        if norm(ΔyTnext, prob.settings[:norm_p]) >
            minimum(ΔyT_norms)

            printstyled("   backtracking"; color=:magenta)
            println()
            println()
            i = 1
            backtrack_yts = []

            iter_ΔyT_norms = []

            while norm(ΔyTnext, prob.settings[:norm_p]) >
                minimum(ΔyT_norms)
                # norm(ΔY.ys[end], prob.settings[:norm_p])
                if i > prob.settings[:max_backtrack_iter]
                    println()
                    printstyled("   max backtracking iterations reached"; color=:magenta)
                    println()
                    println()
                    ΔY = ΔYnext
                    return
                end
                prob.Ẑ = prob.Ẑ - ΔZ
                ΔZ = prob.settings[:α] * ΔZ
                prob.Ẑ = prob.Ẑ + ΔZ
                us = prob.Ẑ.actions
                as = [state[end - 2*udim + 1 : end - udim] for state in prob.Ẑ.states]

                U = hcat(us...)
                A = hcat(as...)

                yTnext = prob.experiment(prob.d2u ? A : U, prob.Ẑ.times; backtracking=true).ys[end]
                ΔyTnext = yTnext - prob.Ygoal.ys[end]

                push!(backtrack_yts, yTnext)
                push!(iter_ΔyT_norms, norm(ΔyTnext, prob.settings[:norm_p]))

                println()
                printstyled("       bt_iter     = ", i; color=:cyan)
                println()
                printstyled("       min |ΔY(T)| = ", minimum(ΔyT_norms); color=:cyan)
                println()
                printstyled("       |ΔY(T)|     = ", norm(ΔyTnext, prob.settings[:norm_p]); color=:cyan)
                println()
                println()

                i += 1
            end
            push!(ΔyT_norms, minimum(iter_ΔyT_norms))
            prob.bt_dict[k] = backtrack_yts

            us = prob.Ẑ.actions
            as = [state[end - 2*udim + 1 : end - udim] for state in prob.Ẑ.states]
            # remeasure with new controls to get full measurement
            U = hcat(us...)
            A = hcat(as...)
            Ȳ = prob.experiment(prob.d2u ? A : U, prob.Ẑ.times)
            ΔY = Ȳ- prob.Ygoal

            # push remeasured norm(ΔyT) to tracked errors
            push!(ΔyT_norms, norm(ΔY.ys[end], prob.settings[:norm_p]))
        else
            ΔY = ΔYnext
            push!(ΔyT_norms, norm(ΔY.ys[end], prob.settings[:norm_p]))
        end
        push!(prob.Ȳs, Ȳ)
        push!(prob.Us, prob.d2u ? A : U)
        k += 1
    end
    @info "ILC converged!" "|ΔY|" = norm(ΔY, prob.settings[:norm_p]) tol = prob.settings[:tol] iter = k
end

# function d2u_to_a(U::Vector{Vector{Float64}}, Δt::Float64)
#     udim = length(U[1])
#     aug_u = fill(zeros(2*udim), length(U))
#     for t = 2:(length(U)-1)
#         aug_u[t] = aug_u[t-1] + (Δt .* [aug_u[t-1][end-udim+1:end]; U[t-1]])
#     end
#     return [a[1:udim] for a in aug_u]
# end

end
