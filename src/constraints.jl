module Constraints

export AbstractConstraint
export EqualityConstraint
export BoundsConstraint
export constrain!

using ..Utils

using Ipopt
using MathOptInterface
const MOI = MathOptInterface

abstract type AbstractConstraint end

struct EqualityConstraint <: AbstractConstraint
    ts::AbstractArray{Int}
    js::AbstractArray{Int}
    vals::Vector{R} where R <: Real
    vardim::Int
end

function EqualityConstraint(
    t::Union{Int, AbstractArray{Int}},
    j::Union{Int, AbstractArray{Int}},
    val::Union{R, Vector{R}},
    vardim::Int
) where R <: Real

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
        vardim
    )
end


function (con::EqualityConstraint)(opt::Ipopt.Optimizer, vars::Vector{MOI.VariableIndex})
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
end

function BoundsConstraint(
    t::Union{Int, AbstractArray{Int}},
    j::Union{Int, AbstractArray{Int}},
    val::Union{Tuple{R, R}, Vector{Tuple{R, R}}},
    vardim::Int
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
        vardim
    )
end

function BoundsConstraint(
    t::Union{Int, AbstractArray{Int}},
    j::Union{Int, AbstractArray{Int}},
    val::Union{R, Vector{R}},
    vardim::Int
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
        vardim
    )
end


function (con::BoundsConstraint)(opt::Ipopt.Optimizer, vars::Vector{MOI.VariableIndex})
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

function constrain!(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    cons::Vector{AbstractConstraint}
)
    for con in cons
        con(opt, vars)
    end
end

end
