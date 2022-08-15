module QuantumSystems

export AbstractQuantumSystem

export QuantumSystem
export TransmonSystem

using ..QuantumLogic

using LinearAlgebra

Im2 = [
    0 -1;
    1  0
]


G(H) = I(2) ⊗ imag(H) - Im2 ⊗ real(H)

abstract type AbstractQuantumSystem end

struct QuantumSystem <: AbstractQuantumSystem
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
    control_bounds::Vector{Float64}
    ψ̃1::Vector{Float64}
    ψ̃goal::Vector{Float64}
    ∫a::Bool
end

function QuantumSystem(
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

        system = QuantumSystem(
            H_drift,
            H_drives,
            ψ1 = ψ1,
            ψf = ψf,
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


function QuantumSystem(
    H_drift::Matrix,
    H_drive::Union{Matrix{T}, Vector{Matrix{T}}},
    ψ1::Union{Vector{C1}, Vector{Vector{C1}}},
    ψf::Union{Vector{C2}, Vector{Vector{C2}}},
    control_bounds::Vector{Float64};
    control_order=2,
    ∫a = false,
    phase = nothing
) where {C1 <: Number, C2 <: Number, T <: Number}

    if !isnothing(phase)
        @assert isa(phase, Float64)
        ψf = exp(1im * phase) * ψf
    end

    if isa(ψ1, Vector{C1})
        nqstates = 1
        isodim = 2 * length(ψ1)
        ψ̃1 = ket_to_iso(ψ1)
        ψ̃goal = ket_to_iso(ψf)
    else
        @assert isa(ψf, Vector{Vector{C2}})
        nqstates = length(ψ1)
        @assert length(ψf) == nqstates
        isodim = 2 * length(ψ1[1])
        ψ̃1 = vcat(ket_to_iso.(ψ1)...)
        ψ̃goal = vcat(ket_to_iso.(ψf)...)
    end

    G_drift = G(H_drift)

    if isa(H_drive, Matrix{T})
        ncontrols = 1
        G_drive = [G(H_drive)]
    else
        ncontrols = length(H_drive)
        G_drive = G.(H_drive)
    end

    @assert length(control_bounds) == length(G_drive)

    augdim = control_order + ∫a

    n_wfn_states = nqstates * isodim
    n_aug_states = ncontrols * augdim

    nstates = n_wfn_states + n_aug_states

    vardim = nstates + ncontrols

    return QuantumSystem(
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
        control_bounds,
        ψ̃1,
        ψ̃goal,
        ∫a
    )
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

end
