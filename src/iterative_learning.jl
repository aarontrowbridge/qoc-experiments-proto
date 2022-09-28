"""
ilqoc.jl -- Iterative learning control for quantum control
"""
module IterativeLearning

export ILCProblem
export solve_ilc!
export discrete_dynamics

using ..Problems
using ..QuantumSystems
using ..Utils
using ..Integrators


using OSQP
using ForwardDiff
using SparseArrays
using LinearAlgebra

include("exponential.jl")

function discrete_dynamics(x, u, sys::QuantumSystem, Δt)
    a = x[sys.n_wfn_states .+ slice(sys.∫a + 1, sys.ncontrols)]
    h_prop = exp(G(a, sys.G_drift, sys.G_drives)*Δt)
    #x_ = zeros(length(x))
    # for i = 1:sys.nqstates
    #     x_[slice(i, sys.isodim)] .= 
    #                 h_prop * x[slice(i, sys.isodim)]
    # end
    states = reduce(vcat, [(h_prop * x[slice(i, sys.isodim)]) for i=1:sys.nqstates])
    
    augs = x[(sys.n_wfn_states + 1):end] + 
    Δt .* [x[(sys.n_wfn_states + 1 + sys.ncontrols):end]; u]
        

    return [states; augs]

end

#gs return measurement value given quantum state

# y = g(x)

#exp_rollout takes in utraj and returns ys @ ts.
struct ILCProblem 
    n::Int
    m::Int
    d::Int
    T::Int
    Δt::Float64
    g::Function
    ỹs::Vector{Vector{Float64}}
    exp_rollout::Function
    xnom::Matrix{Float64}
    unom::Matrix{Float64}
    utraj::Matrix{Float64}
    ctraj::Matrix{Float64}
    control_bounds::Vector{Float64}
    A::Vector{SparseArrays.SparseMatrixCSC{Float64, Int64}}
    B::Vector{SparseArrays.SparseMatrixCSC{Float64, Int64}}
    C::Vector{SparseArrays.SparseMatrixCSC{Float64, Int64}}
    Cinv::Vector{SparseArrays.SparseMatrixCSC{Float64, Int64}}
end

function ILCProblem(
    nominal_prob::Union{FixedTimeProblem, MinTimeProblem},
    g::Function,
    exp_rollout::Function,
    d::Int;
    ỹs=nothing
)
    println("Solving nominal problem...")
    solve!(nominal_prob)
    traj = nominal_prob.trajectory
    sys = nominal_prob.system

    Xopt = traj.states
    Uopt = traj.actions

    n = sys.nstates
    m = sys.ncontrols

    T = traj.T
    Δt = traj.Δt
    xnom = zeros(n, T)
    
    for k = 1:T
        xnom[:, k] .= Xopt[k]
    end
    
    unom = zeros(m, T-1)
    utraj = zeros(m, T-1)

    for k = 1:(T-1)
        unom[:, k] .= Uopt[k]
        utraj[:, k] .= Uopt[k] 
    end

    
    if isnothing(ỹs)
        ỹs = [g(x, sys) for x in Xopt]
    end
    
    A = fill(spzeros(n, n), T-1)
    B = fill(spzeros(n, m), T-1)
    C = fill(spzeros(d, n),T)
    Cinv = fill(spzeros(n, d), T)

    for k = 1:(T-1)
        A[k] .= sparse(ForwardDiff.jacobian(x->discrete_dynamics(x, unom[:,k], sys, traj.Δt), xnom[:,k]))
        B[k] .= sparse(ForwardDiff.jacobian(u->discrete_dynamics(xnom[:,k], u, sys, traj.Δt), unom[:,k]))
    end 

    for k = 1:T
        C[k] .= sparse(ForwardDiff.jacobian(x -> g(x, sys), xnom[:,k]))
        Cinv[k]  .= sparse(pinv(Matrix(C[k])))
    end

    ctraj = zeros(2*sys.ncontrols, T)
    for k = 1:T
        ctraj[:, k] .= xnom[end - 3:end, k]
    end
    
    return ILCProblem(
        n,
        m,
        d,
        T,
        Δt,
        g,
        ỹs,
        exp_rollout,
        xnom,
        unom,
        utraj,
        ctraj,
        sys.control_bounds,
        A, 
        B,
        C,
        Cinv
    )
    
end

function solve_ilc!(
    ilc::ILCProblem; 
    iter=3,
    tol=1e-4, 
    qs=[fill(1., ilc.n - 2*ilc.m); zeros(2*ilc.m)],
    qfs=[fill(8000., ilc.n - 2*ilc.m); zeros(2*ilc.m)],
    R=0.01
    )   
    
    #answer, adj = print(ilc.exp_rollout(ilc.utraj))

    #print(answer[end])

    @assert length(qs) == ilc.n
    @assert length(qfs) == ilc.n

    #unpack the struct for convenience
    n = ilc.n
    m = ilc.m
    d = ilc.d
    T= ilc.T


    Q = sparse(Diagonal(qs))
    Qf = sparse(Diagonal(qfs))
    R = sparse(Diagonal(fill(R, m)))
    S = sparse(Diagonal(fill(1,d)))
    Sf = sparse(Diagonal(fill(8000., d)))

    S = spzeros(d,d)
    A = ilc.A
    B = ilc.B
    C = ilc.C
    Cinv = ilc.Cinv
    
    diag = map(x -> [R, Q, x'*Q*x], Cinv)
    diag = reduce(vcat, diag)
    deleteat!(diag, (length(diag) - 2):length(diag))
    deleteat!(diag, 1:3)
    #H = blockdiag(kron(I(T-2), blockdiag(R, Q, S)), R, Qf, Sf)

    H = blockdiag(diag..., R, Qf, Cinv[end]' * Qf * Cinv[end])
    #total dimension of our vector is (n+m+d)*(T-1)
    
    #dynamics constraints
    D = spzeros(n*(T-1), (n+m+d)*(T-1))
    D[1:n, 1:m] .= B[1]
    D[1:n, m .+ (1:n)] .= -I(n)

    for k = 1:(T-2)
        D[k*n .+ (1:n), 
          ((k-1)*(n+d+m) + m) .+ (1:(2*n + m + d))
         ] .= [A[k+1] spzeros(n, d) B[k+1] -I]
    end

    #measurement Constraints
    M = spzeros(d*(T-1), (n+m+d)*(T-1))

    for k = 1:(T-1)
        M[(k-1)*d .+ (1:d), (m + (k-1)*(n+d+m)) .+ (1:(n+d))] .= [C[k+1] -I]
    end
    
    #control bound constraints
    Cb = kron(I(T-1), [zeros(m, n - m) I zeros(m, d + m)])

    #matrix that picks out the Δus
    U = kron(I(T-1), [I zeros(m, n + d)])

    
    it = 0
    while it < iter
        ys, ts = ilc.exp_rollout(ilc.utraj)
        @assert all(t -> 1 <= t <= T, ts)
        @assert length(ys) == length(ts)

        println("Iter $(it): $(ys[end])")

        q = zeros((n+m+d)*(T-1))

        
        errs = zeros(d*(T-1))
        
        for (y, t) in zip(ys, ts)
            errs[d*(t-2) .+ (1:d)] .= -(y - ilc.ỹs[t])
        end

        #println(errs[end])
        #S, Sf
        for k = 1:T-2
            q[(m+n) + (m+n+d)*(k-1) .+ (1:d)] .= Cinv[k+1]'*Q*Cinv[k+1]*(-errs[(k-1)*d .+ (1:d)])
        end

        q[(m+n) + (m+n+d)*(T-2) .+ (1:d)] .= Cinv[end]'*Qf*Cinv[end]*(-errs[(T-2)*d .+ (1:d)])
        

        cbnds = [repeat(ilc.control_bounds, T-2); 
                zeros(length(ilc.control_bounds))]
        cb_traj = reduce(vcat, ilc.ctraj[1:m, 2:end])

        lb = [zeros(n*(T-1)); errs; -cbnds-cb_traj]
        ub = [zeros(n*(T-1)); errs; cbnds-cb_traj]
        qp = OSQP.Model()
        OSQP.setup!(qp, P=H, q=q, A=[D; M; Cb], l=lb, u=ub, 
                    eps_abs=1e-8, eps_rel=1e-8, eps_prim_inf=1e-8, 
                    eps_dual_inf=1e-8, polish=1)
        results = OSQP.solve!(qp)
        ztraj = results.x
        #print(U*ztraj)
        Δu = reshape(U*ztraj, 2, :)
        
        ilc.utraj .= ilc.utraj + Δu

        for k = 2:T
            ilc.ctraj[:, k] .= ilc.ctraj[:, k-1] + ilc.Δt .* 
            [ilc.ctraj[end - m + 1:end, k-1]; ilc.utraj[:, k-1]]
        end
        it += 1
    end

    return ilc.exp_rollout(ilc.utraj)   

end



end