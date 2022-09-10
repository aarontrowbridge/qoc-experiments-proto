module Constraints

export problem_constraints
export constrain!

export AbstractConstraint

export L1SlackConstraint
export EqualityConstraint
export BoundsConstraint
export TimeStepBoundsConstraint

using ..Utils
using ..QuantumSystems

using Ipopt
using MathOptInterface
const MOI = MathOptInterface


abstract type AbstractConstraint end

function constrain!(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    cons::Vector{AbstractConstraint};
    verbose=false
)
    for con in cons
        if verbose
            println("applying constraint: ", con.name)
        end
        con(opt, vars)
    end
end


struct EqualityConstraint <: AbstractConstraint
    ts::AbstractArray{Int}
    js::AbstractArray{Int}
    vals::Vector{R} where R
    vardim::Int
    name::String
end

function EqualityConstraint(
    t::Union{Int, AbstractArray{Int}},
    j::Union{Int, AbstractArray{Int}},
    val::Union{R, Vector{R}},
    vardim::Int;
    name="unnamed equality constraint"
) where R

    @assert !(isa(val, Vector{R}) && isa(j, Int))
        "if val is an array, j must be an array of integers"

    @assert isa(val, R) ||
        (isa(val, Vector{R}) && isa(j, AbstractArray{Int})) &&
        length(val) == length(j) """
    if j and val are both arrays, dimensions must match:
        length(j)   = $(length(j))
        length(val) = $(length(val))
    """

    if isa(val, R) && isa(j, AbstractArray{Int})
        val = fill(val, length(j))
    end

    return EqualityConstraint(
        [t...],
        [j...],
        [val...],
        vardim,
        name
    )
end


function (con::EqualityConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex}
)
    for t in con.ts
        for (j, val) in zip(con.js, con.vals)
            MOI.add_constraints(
                opt,
                vars[index(t, j, con.vardim)],
                MOI.EqualTo(val)
            )
        end
    end
end

struct BoundsConstraint <: AbstractConstraint
    ts::AbstractArray{Int}
    js::AbstractArray{Int}
    vals::Vector{Tuple{R, R}} where R <: Real
    vardim::Int
    name::String
end

function BoundsConstraint(
    t::Union{Int, AbstractArray{Int}},
    j::Union{Int, AbstractArray{Int}},
    val::Union{Tuple{R, R}, Vector{Tuple{R, R}}},
    vardim::Int;
    name="unnamed bounds constraint"
) where R <: Real

    @assert !(isa(val, Vector{Tuple{R, R}}) && isa(j, Int))
        "if val is an array, var must be an array of integers"

    if isa(val, Tuple{R,R}) && isa(j, AbstractArray{Int})

        val = fill(val, length(j))

    elseif isa(val, Tuple{R, R}) && isa(j, Int)

        val = [val]
        j = [j]

    end

    @assert *([v[1] <= v[2] for v in val]...) "lower bound must be less than upper bound"

    return BoundsConstraint(
        [t...],
        j,
        val,
        vardim,
        name
    )
end

function BoundsConstraint(
    t::Union{Int, AbstractArray{Int}},
    j::Union{Int, AbstractArray{Int}},
    val::Union{R, Vector{R}},
    vardim::Int;
    name="unnamed bounds constraint"
) where R <: Real

    @assert !(isa(val, Vector{R}) && isa(j, Int))
        "if val is an array, var must be an array of integers"

    if isa(val, R) && isa(j, AbstractArray{Int})

        bounds = (-abs(val), abs(val))
        val = fill(bounds, length(j))

    elseif isa(val, R) && isa(j, Int)

        bounds = (-abs(val), abs(val))
        val = [bounds]
        j = [j]

    elseif isa(val, Vector{R})

        val = [(-abs(v), abs(v)) for v in val]

    end

    return BoundsConstraint(
        [t...],
        j,
        val,
        vardim,
        name
    )
end

function (con::BoundsConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex}
)
    for t in con.ts
        for (j, (lb, ub)) in zip(con.js, con.vals)
            MOI.add_constraints(
                opt,
                vars[index(t, j, con.vardim)],
                MOI.GreaterThan(lb)
            )
            MOI.add_constraints(
                opt,
                vars[index(t, j, con.vardim)],
                MOI.LessThan(ub)
            )
        end
    end
end

struct TimeStepBoundsConstraint <: AbstractConstraint
    bounds::Tuple{R, R} where R <: Real
    T::Int
    name::String
    function TimeStepBoundsConstraint(
        bounds::Tuple{R, R} where R <: Real,
        T::Int;
        name="unnamed time step bounds constraint"
    )
        @assert bounds[1] < bounds[2] "lower bound must be less than upper bound"
        return new(bounds, T, name)
    end
end

function (con::TimeStepBoundsConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex}
)
    for t = 1:(con.T - 1)
        MOI.add_constraints(
            opt,
            vars[end - (con.T - 1) + t],
            MOI.GreaterThan(con.bounds[1])
        )
        MOI.add_constraints(
            opt,
            vars[end - (con.T - 1) + t],
            MOI.LessThan(con.bounds[2])
        )
    end
end

struct L1SlackConstraint <: AbstractConstraint
    s1_indices::AbstractArray{Int}
    s2_indices::AbstractArray{Int}
    x_indices::AbstractArray{Int}
    name::String

    function L1SlackConstraint(
        s1_indices::AbstractArray{Int},
        s2_indices::AbstractArray{Int},
        x_indices::AbstractArray{Int};
        name="unmamed L1 slack constraint"
    )
        @assert length(s1_indices) == length(s2_indices) == length(x_indices)
        return new(s1_indices, s2_indices, x_indices, name)
    end
end

function (con::L1SlackConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex}
)
    for (s1, s2, x) in zip(
        con.s1_indices,
        con.s2_indices,
        con.x_indices
    )
        MOI.add_constraints(
            opt,
            vars[s1],
            MOI.GreaterThan(0.0)
        )
        MOI.add_constraints(
            opt,
            vars[s2],
            MOI.GreaterThan(0.0)
        )
        t1 = MOI.ScalarAffineTerm(1.0, vars[s1])
        t2 = MOI.ScalarAffineTerm(-1.0, vars[s2])
        t3 = MOI.ScalarAffineTerm(-1.0, vars[x])
        MOI.add_constraints(
            opt,
            MOI.ScalarAffineFunction([t1, t2, t3], 0.0),
            MOI.EqualTo(0.0)
        )
    end
end


function problem_constraints(
    system::AbstractSystem,
    T::Int;
    pin_first_qstate=false
)
    cons = AbstractConstraint[]

    # initial quantum state constraints: ψ̃(t=1) = ψ̃1
    ψ1_con = EqualityConstraint(
        1,
        1:system.n_wfn_states,
        system.ψ̃init,
        system.vardim;
        name="initial quantum state constraints"
    )
    push!(cons, ψ1_con)

    # initial a(t = 1) constraints: ∫a, a, da = 0
    aug_cons = EqualityConstraint(
        [1, T],
        system.n_wfn_states .+ (1:system.n_aug_states),
        0.0,
        system.vardim;
        name="initial and final augmented state constraints"
    )
    push!(cons, aug_cons)

    # bound |a(t)| < a_bound
    @assert length(system.control_bounds) ==
        length(system.G_drives)

    for cntrl_index in 1:system.ncontrols
        cntrl_bound = BoundsConstraint(
            2:T-1,
            system.n_wfn_states +
            system.∫a * system.ncontrols +
            cntrl_index,
            system.control_bounds[cntrl_index],
            system.vardim;
            name="constraint on control $(cntrl_index)"
        )
        push!(cons, cntrl_bound)
    end

    # pin first qstate to be equal to analytic solution
    if pin_first_qstate
        ψ̃¹goal = system.ψ̃goal[1:system.isodim]
        pin_con = EqualityConstraint(
            T,
            1:system.isodim,
            ψ̃¹goal,
            system.vardim;
            name="pinned first qstate at T"
        )
        push!(cons, pin_con)
    end

    return cons
end


end
