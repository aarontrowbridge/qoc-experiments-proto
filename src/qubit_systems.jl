module QubitSystems

export AbstractQuantumSystem

export SingleQubitSystem
export MultiModeQubitSystem
export TransmonSystem
export TwoQubitSystem

using ..QuantumLogic

using LinearAlgebra
using HDF5

Im2 = [
    0 -1;
    1  0
]


G(H) = I(2) ⊗ imag(H) - Im2 ⊗ real(H)

abstract type AbstractQuantumSystem end

struct SingleQubitSystem <: AbstractQuantumSystem
    n_wfn_states::Int
    n_aug_states::Int
    nstates::Int
    nqstates::Int
    isodim::Int
    augdim::Int
    vardim::Int
    ncontrols::Int
    control_order::Int
    G_drift::Matrix{Float64}
    G_drives::Vector{Matrix{Float64}}
    ψ̃1::Vector{Float64}
    ψ̃goal::Vector{Float64}
    gate::Union{Symbol, Nothing}
    ∫a::Bool
end

function SingleQubitSystem(
    H_drift::Matrix,
    H_drive::Union{Matrix{T}, Vector{Matrix{T}}},
    gate::Union{Symbol, Nothing},
    ψ1::Union{Vector{C}, Vector{Vector{C}}};
    ψf=nothing,
    control_order=2,
    ∫a = true
) where {C <: Number, T <: Number}

    if isa(ψ1, Vector{C})
        nqstates = 1
        isodim = 2 * length(ψ1)
        if isnothing(ψf)
            ψ̃goal = ket_to_iso(apply(gate, ψ1))
        else
            ψ̃goal = ket_to_iso(ψf)
        end
        ψ̃1 = ket_to_iso(ψ1)
    else
        nqstates = length(ψ1)
        isodim = 2 * length(ψ1[1])
        if isnothing(ψf)
            ψ̃goal = vcat(ket_to_iso.(apply.(gate, ψ1))...)
        else
            @assert isa(ψf, Vector{Vector{C}})
            ψ̃goal = vcat(ket_to_iso.(ψf)...)
        end
        ψ̃1 = vcat(ket_to_iso.(ψ1)...)
    end

    G_drift = G(H_drift)

    if isa(H_drive, Matrix{T})
        ncontrols = 1
        G_drive = [G(H_drive)]
    else
        ncontrols = length(H_drive)
        G_drive = G.(H_drive)
    end

    augdim = control_order + ∫a

    n_wfn_states = nqstates * isodim
    n_aug_states = ncontrols * augdim

    nstates = n_wfn_states + n_aug_states

    vardim = nstates + ncontrols

    return SingleQubitSystem(
        n_wfn_states,
        n_aug_states,
        nstates,
        nqstates,
        isodim,
        augdim,
        vardim,
        ncontrols,
        control_order,
        G_drift,
        G_drive,
        ψ̃1,
        ψ̃goal,
        gate,
        ∫a
    )
end

function SingleQubitSystem(
    H_drift::Matrix,
    H_drive::Union{Matrix{C}, Vector{Matrix{C}}},
    ψ1::Union{Vector{C}, Vector{Vector{C}}},
    ψf::Union{Vector{C}, Vector{Vector{C}}};
    kwargs...
) where C <: Number
    return SingleQubitSystem(
        H_drift,
        H_drive,
        nothing,
        ψ1;
        ψf=ψf,
        kwargs...
    )
end


#
# Multi-mode Qubit System
#

struct MultiModeQubitSystem <: AbstractQuantumSystem
    ncontrols::Int
    nqstates::Int
    nstates::Int
    n_wfn_states::Int
    n_aug_states::Int
    isodim::Int
    augdim::Int
    vardim::Int
    control_order::Int
    G_drift::Matrix{Float64}
    G_drives::Vector{Matrix{Float64}}
    ψ̃1::Vector{Float64}
    ψ̃goal::Vector{Float64}
    ∫a::Bool
end

function MultiModeQubitSystem(
    H_drift::Matrix,
    H_drives::Vector{Matrix{C}} where C,
    ψ1::Vector,
    ψgoal::Vector;
    control_order=2,
    ∫a = false
)
    isodim = 2 * length(ψ1)

    ψ̃1 = ket_to_iso(ψ1)
    ψ̃goal = ket_to_iso(ψgoal)

    G_drift = G(H_drift)

    ncontrols = length(H_drives)

    G_drives = G.(H_drives)

    nqstates = 1

    n_wfn_states = nqstates * isodim

    augdim = control_order + ∫a

    n_aug_states = ncontrols * augdim

    nstates = n_wfn_states + n_aug_states

    vardim = nstates + ncontrols

    return MultiModeQubitSystem(
        ncontrols,
        nqstates,
        nstates,
        n_wfn_states,
        n_aug_states,
        isodim,
        augdim,
        vardim,
        control_order,
        G_drift,
        G_drives,
        ψ̃1,
        ψ̃goal,
        ∫a
    )
end

function MultiModeQubitSystem(
    hf_path::String;
    return_data=false,
    kwargs...
)
    h5open(hf_path, "r") do hf

        H_drift = hf["H_drift"][:, :]

        H_drives = [
            copy(transpose(hf["H_drives"][:, :, i]))
                for i = 1:size(hf["H_drives"], 3)
        ]

        ψ1 = vcat(transpose(hf["psi1"][:, :])...)
        ψf = vcat(transpose(hf["psif"][:, :])...)

        system = MultiModeQubitSystem(
            H_drift,
            H_drives,
            ψ1,
            ψf;
            kwargs...
        )

        if return_data

            data = Dict()

            controls = copy(transpose(hf["controls"][:, :]))
            data["controls"] = controls

            ts = hf["tlist"][:]
            data["T"] = length(ts)
            data["Δt"] = ts[2] - ts[1]

            return system, data
        else
            return system
        end
    end
end

struct TransmonSystem <: AbstractQuantumSystem
    n_wfn_states::Int
    n_aug_states::Int
    nstates::Int
    nqstates::Int
    isodim::Int
    augdim::Int
    vardim::Int
    ncontrols::Int
    control_order::Int
    G_drift::Matrix{Float64}
    G_drives::Vector{Matrix{Float64}}
    ψ̃1::Vector{Float64}
    ψ̃goal::Vector{Float64}
    ∫a::Bool
end

function TransmonSystem(;
    levels::Int,
    rotating_frame::Bool,
    ω::Float64,
    α::Float64,
    ψ1::Union{Vector{C}, Vector{Vector{C}}},
    ψf::Union{Vector{C}, Vector{Vector{C}}},
    control_order = 2,
    ∫a = false
) where C <: Number
    # if it is just one state
    if isa(ψ1, Vector{C})
        nqstates = 1
        isodim = 2 * length(ψ1)
        ψ̃goal = ket_to_iso(ψf)
        ψ̃1 = ket_to_iso(ψ1)
    # otherwise it is multiple states and we are defining a (partial) isometry
    else
        nqstates = length(ψ1)
        isodim = 2 * length(ψ1[1])
        @assert isa(ψf, Vector{Vector{C}})
        # takes care of real-to-complex isomorphism and stacks the states
        ψ̃goal = vcat(ket_to_iso.(ψf)...)
        ψ̃1 = vcat(ket_to_iso.(ψ1)...)
    end

    if rotating_frame
        H_drift = α/2 * quad(levels)
    else
        H_drift = ω * number(levels) + α/2 * quad(levels)
    end

    G_drift = G(H_drift)

    ncontrols = 2

    H_drive = [create(levels) + annihilate(levels),
              1im * (create(levels) - annihilate(levels))]
    G_drive = G.(H_drive)


    augdim = control_order + ∫a

    n_wfn_states = nqstates * isodim
    n_aug_states = ncontrols * augdim

    nstates = n_wfn_states + n_aug_states

    vardim = nstates + ncontrols

    return TransmonSystem(
        n_wfn_states,
        n_aug_states,
        nstates,
        nqstates,
        isodim,
        augdim,
        vardim,
        ncontrols,
        control_order,
        G_drift,
        G_drive,
        ψ̃1,
        ψ̃goal,
        ∫a
    )
end


struct TwoQubitSystem <: AbstractQuantumSystem
    n_wfn_states::Int
    n_aug_states::Int
    nstates::Int
    nqstates::Int
    isodim::Int
    augdim::Int
    vardim::Int
    ncontrols::Int
    control_order::Int
    G_drift::Matrix{Float64}
    G_drives::Vector{Matrix{Float64}}
    ψ̃1::Vector{Float64}
    ψ̃goal::Vector{Float64}
    ∫a::Bool
end

function TwoQubitSystem(;
    H_drift::Matrix, 
    H_drive::Union{Matrix{T}, Vector{Matrix{T}}},
    ψ1::Union{Vector{C}, Vector{Vector{C}}},
    ψf::Union{Vector{C}, Vector{Vector{C}}},
    control_order = 2,
    ∫a = false
) where {C <: Number, T <: Number}
    if isa(ψ1, Vector{C})
        nqstates = 1
        isodim = 2 * length(ψ1)
        ψ̃goal = ket_to_iso(ψf)
        ψ̃1 = ket_to_iso(ψ1)
    else
        nqstates = length(ψ1)
        isodim = 2 * length(ψ1[1])
        @assert isa(ψf, Vector{Vector{C}})
        # takes care of real-to-complex isomorphism and stacks the states
        ψ̃goal = vcat(ket_to_iso.(ψf)...)
        ψ̃1 = vcat(ket_to_iso.(ψ1)...)
    end

    ncontrols = 2

    # H_drift = -(ω1/2 + gcouple)*kron(GATES[:Z], I(2)) -
    #            (ω2/2 + gcouple)*kron(I(2), GATES[:Z]) +
    #            gcouple*kron(GATES[:Z], GATES[:Z])

    # H_drift = J*kron(GATES[:Z], GATES[:Z])
    H_drift = zeros(4,4) #ω1/2 * kron(GATES[:Z], I(2)) + ω2/2 * kron(I(2), GATES[:Z])
    G_drift = G(H_drift)

    if isa(H_drive, Matrix{T})
        ncontrols = 1
        G_drive = [G(H_drive)]
    else
        ncontrols = length(H_drive)
        G_drive = G.(H_drive)
    end

    augdim = control_order + ∫a

    n_wfn_states = nqstates * isodim
    n_aug_states = ncontrols * augdim

    nstates = n_wfn_states + n_aug_states

    vardim = nstates + ncontrols

    return TwoQubitSystem(
        n_wfn_states,
        n_aug_states,
        nstates,
        nqstates,
        isodim,
        augdim,
        vardim,
        ncontrols,
        control_order,
        G_drift,
        G_drive,
        ψ̃1,
        ψ̃goal,
        ∫a
    )

end

end
