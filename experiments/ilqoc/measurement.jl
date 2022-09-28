"""
measurement.jl - functions for doing measurement
"""

#Single Qubit

using LinearAlgebra
using Random
using Distributions

include("exponential.jl")

#
# Noisy Operators and Measurements
#

function sigmaz()
    return [1 0; 
            0 -1]
end

function sigmax()
    return [0 1;
            1 0]
end

function sigmay()
    return [0 -1im;
            1im 0]
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

function meas_x(state::Vector{<:Number})
    real(state' * sigmax() * state)
end

function meas_x_iso(iso_state::Vector{<:Real})
    iso_state' * get_mat_iso(sigmax()) * iso_state
end

function meas_y(state::Vector{<:Number})
    real(state' * sigmay() * state)
end

function meas_y_iso(iso_state::Vector{<:Real})
    iso_state' * get_mat_iso(sigmay()) * iso_state
end

function meas_x_noisy(state::Vector{<:Number}; error = 0.025)
    #might need to take real part cause of numerical error
    real(state' * noisy_sigmax(error = error) * state)
end

function meas_x_noisy_iso(iso_state::Vector{<:Real}; error = 0.025)
    #might need to take real part cause of numerical error
    iso_state' * get_mat_iso(noisy_sigmax(error = error)) * iso_state
end


function meas_y_noisy(state::Vector{<:Number}; error = 0.025)
    real(state' * noisy_sigmay(error = error) * state)
end

function meas_y_noisy_iso(iso_state::Vector{<:Real}; error = 0.025)
    #might need to take real part cause of numerical error
    iso_state' * get_mat_iso(noisy_sigmay(error = error)) * iso_state
end

function meas_z(state::Vector{<:Number})
    real(state' * sigmaz() * state)
end

function meas_z_iso(iso_state::Vector{<:Real})
    iso_state' * get_mat_iso(sigmaz()) * iso_state
end


function pauli_meas_g(state::Vector{<:Number}; error = 0.025, acquisition_num = 1000, readout_fidelity = 0.96)
    running_x = 0 
    running_y = 0 
    running_z = 0
    
    for i in 1:acquisition_num
        running_x += sign(2 * rand() - 1 + 
                          (1-2*(1 - readout_fidelity) * meas_x_noisy(state, error = error)))
        running_y += sign(2 * rand() - 1 + 
                          (1-2*(1 - readout_fidelity) * meas_y_noisy(state, error = error)))
        running_z += sign(2 * rand() - 1 + 
                          (1-2*(1 - readout_fidelity) * meas_z(state)))
    end
    return [running_x/acquisition_num, running_y/acquisition_num, running_z/acquisition_num]
end

function pauli_meas_g_iso(iso_state::Vector{<:Real}; error = 0.025, acquisition_num = 1000, readout_fidelity = 0.96)
    running_x = 0 
    running_y = 0 
    running_z = 0
    
    for i in 1:acquisition_num
        running_x += sign(2 * rand() - 1 + 
                          (1-2*(1 - readout_fidelity) * meas_x_noisy_iso(iso_state, error = error)))
        running_y += sign(2 * rand() - 1 + 
                          (1-2*(1 - readout_fidelity) * meas_y_noisy_iso(iso_state, error = error)))
        running_z += sign(2 * rand() - 1 + 
                          (1-2*(1 - readout_fidelity) * meas_z_iso(iso_state)))
    end
    return [running_x/acquisition_num, running_y/acquisition_num, running_z/acquisition_num]
end
