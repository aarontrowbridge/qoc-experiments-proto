"""
qctest.jl - testing QuantumCumulants.jl
"""

using QuantumCumulants
using OrdinaryDiffEq, ModelingToolkit
using Plots

@syms t::Real


@register r(t)
@register i(t)

hq = NLevelSpace(:qubit,(:g,:e))
σ = Transition(hq, :σ)
# σz = σ(:g, :g) - σ(:e, :e)
# σy = 1im*(σ(:e, :g)-σ(:g, :e))
# σx = σ(:g, :e) + σ(:e, :g)

#Hamiltonian 

H = r(t) * (σ(:e, :g) + σ(:g, :e)) + i(t)*1im*(σ(:e, :g) - σ(:g, :e))

eqns = meanfield([σ(:g, :e), σ(:e, :g), σ(:g, :g), σ(:e, :e)], H)

@named sys = ODESystem(eqns)


function r(t)
    return 2π*19e-3
end

function i(t)
    return 0.
end

u0 = [0 + 0im, 0. + 0im, 1. + 0im, 0 + 0im]

prob = ODEProblem(sys, u0, (0.0, 1/(4*19e-3)))

sol = solve(prob, Tsit5())

t = sol.t
x = sol.u

# s22 = real.(getindex.(x,3) - getindex.(x,4))
# display(s22)

# plt = plot(t, s22, xlabel="t", ylabel="⟨σz⟩", legend=false, size=(600,300))
# savefig(plt, "test.png")