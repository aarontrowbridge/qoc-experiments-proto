"""
measurement.jl - functions for doing measurement on various SystemDynamics
"""

#Single Qubit

using LinearAlgebra
using Random
using Distributions

#
# Noisy Operators and Measurements
#

function sigmaz()
    return [1 0; 
            0 -1]
end

function noisy_sigmax(;error = 0.025)
    H = noisy_hadamard(error = error)
    return H*sigmaz()*H
end

function noisy_sigmay(; error = 0.025)
    HS_dag = noisy_hadamard(error = error) * S_dag()
    return HS_dag' * sigmaz() * HS_dag
end

function S_dag()
    return [1 0;
            0 -1im]
end

function noisy_hadamard(;error = 0.025)
    d = Normal(π/2, π/2 * error)
    θ =  rand(d)
    return [cos(θ/2) sin(θ/2);
            sin(θ/2) -cos(θ/2)]
end

function meas_x_noisy(state, error)
    2
end
function pauli_meas_g(state; error = 0.025, acquisition_num = 1000, readout_fidelity = 0.96)
    
end

