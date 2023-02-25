module QuantumSystems

export AbstractSystem

export QuantumSystem
export MultiModeSystem
export TransmonSystem

using ..QuantumUtils

using HDF5

using LinearAlgebra

Im2 = [
    0 -1;
    1  0
]

G(H) = I(2) ⊗ imag(H) - Im2 ⊗ real(H)

"""
```julia
AbstractSystem
```

Abstract type for defining systems.
"""
abstract type AbstractSystem end

# TODO: make subtypes: SingleQubitSystem, TwoQubitSystem, TransmonSystem, MultimodeSystem, etc.

"""
```julia
QuantumSystem <: AbstractSystem
```

Defines a struct that contains all the information and parameters pertinent to a quantum system.
"""
struct QuantumSystem <: AbstractSystem
    n_wfn_states::Int
    n_aug_states::Int
    nstates::Int
    nqstates::Int
    isodim::Int
    ketdim::Int
    augdim::Int
    vardim::Int
    ncontrols::Int
    control_order::Int
    G_drift::Matrix{Float64}
    G_drives::Vector{Matrix{Float64}}
    a_bounds::Vector{Float64}
    ψ̃init::Vector{Float64}
    ψ̃goal::Vector{Float64}
    ∫a::Bool
    params::Dict{Symbol, Any}
end


"""
    QuantumSystemNew

A struct for storing the isomorphisms of the system's drift and drive Hamiltonians,
as well as the system's parameters.
"""
struct QuantumSystemNew <: AbstractSystem
    G_drift::Matrix{Float64}
    G_drives::Vector{Matrix{Float64}}
    params::Dict{Symbol, Any}
end

"""
   QuantumSystemNew(
        H_drift::Matrix{<:Number},
        H_drives::Vector{Matrix{<:Number}};
        params = Dict{Symbol, Any}(),
        kwargs...
   )

Constructs a `QuantumSystemNew` object from the drift and drive Hamiltonians.
"""
function QuantumSystemNew(
    H_drift::Matrix{<:Number},
    H_drives::Vector{Matrix{<:Number}};
    params = Dict{Symbol, Any}(),
    kwargs...
)
    G_drift = G(H_drift)
    G_drives = G.(H_drives)
    params = merge(params, Dict(kwargs...))
    return QuantumSystemNew(G_drift, G_drives, params)
end


# TODO: move ψinit and ψgoal into prob def
"""
```julia
QuantumSystem(
    H_drift::Matrix,
    H_drive::Union{Matrix{T}, Vector{Matrix{T}}},
    ψinit::Union{Vector{C1}, Vector{Vector{C1}}},
    ψgoal::Union{Vector{C2}, Vector{Vector{C2}}},
    control_bounds::Vector{Float64};
    ∫a=false,
    control_order=2,
    goal_phase=0.0
) where {C1 <: Number, C2 <: Number, T <: Number}
```

* `H_drift`: drift Hamiltonian
* `H_drive`: drive Hamiltonian(s)
* `ψinit`: initial state(s)
* `ψgoal`: goal state(s)
* `control_bounds`: bounds on control amplitudes
* `∫a`: whether to include the integral of the control varialbes in the augmented state - defaults to `false`
* `control_order`: the control derivative order which functions as the actual decision variable - defaults to `2` (i.e. the decision variable is the second derivative: ``\\ddot a(t)``)
* `goal_phase`: the phase of the initial guess goal state(s) (which is linearly interpolated from the initial state when problem is cold started) - defaults to `0.0`
"""
function QuantumSystem(
    H_drift::Matrix,
    H_drives::Vector{Matrix{T}},
    ψinit::Union{Vector{C1}, Vector{Vector{C1}}},
    ψgoal::Union{Vector{C2}, Vector{Vector{C2}}};
    a_bounds::Vector{Float64}=fill(Inf, length(H_drives)),
    ∫a=false,
    control_order=2,
    goal_phase=0.0,
    params=Dict()
) where {C1 <: Number, C2 <: Number, T <: Number}

    params[:goal_phase] = goal_phase

    if isa(ψinit, Vector{C1})
        nqstates = 1
        ketdim = length(ψinit)
        isodim = 2 * ketdim
        ψ̃init = ket_to_iso(ψinit)
        ψgoal *= exp(im * goal_phase)
        ψ̃goal = ket_to_iso(ψgoal)
    else
        @assert isa(ψgoal, Vector{Vector{C2}})
        nqstates = length(ψinit)
        @assert length(ψgoal) == nqstates
        ketdim = length(ψinit[1])
        isodim = 2 * ketdim
        ψ̃init = vcat(ket_to_iso.(ψinit)...)
        ψgoal[1] *= exp(im * goal_phase)
        ψ̃goal = vcat(ket_to_iso.(ψgoal)...)
    end

    G_drift = G(H_drift)

    ncontrols = length(H_drives)
    G_drives = G.(H_drives)

    @assert length(a_bounds) == ncontrols

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
        ketdim,
        augdim,
        vardim,
        ncontrols,
        control_order,
        G_drift,
        G_drives,
        a_bounds,
        ψ̃init,
        ψ̃goal,
        ∫a,
        params
    )
end


"""
```julia
MultiModeSystem(
    transmon_levels::Int,
    cavity_levels::Int,
    ψ1::String, # e.g. "g0" or "e3" or "eN"
    ψf::String; # e.g. "g1" or "e4" or "eN";
    χ=2π * -0.5459e-3,
    κ=2π * 4e-6,
    transmon_control_bound=2π * 0.018,
    cavity_control_bound=0.03,
    n_cavities=1, # TODO: add functionality for multiple cavities
    kwargs...
)
```

* `transmon_levels`: available levels for transmon qubit
* `cavity_levels`: available levels for cavity
* `ψ1`: initial state as a string (e.g. "g0", "e3", ..., or "eN")
* `ψf`: final state as a string (e.g. "g1", "e4", ..., or "eN")
* `χ`: transmon-cavity coupling strength - defaults to `2π * -0.5459e-3`
* `κ`: cavity decay rate - defaults to `2π * 4e-6`
* `transmon_control_bound`: bound on transmon control amplitude - defaults to `2π * 0.018`
* `cavity_control_bound`: bound on cavity control amplitude - defaults to `0.03`
* `n_cavities`: number of cavities - defaults to `1`
* `kwargs...`: keyword arguments to pass to the `QuantumSystem` constructor function
"""
function MultiModeSystem(
    transmon_levels::Int,
    cavity_levels::Int,
    ψ1::String, # e.g. "g0" or "e3" or "eN"
    ψf::String; # e.g. "g1" or "e4" or "eN";
    χ=2π * -0.5459e-3,
    κ=2π * 4e-6,
    χGF=2π * -1.01540302914e-3,
    α=-2π * 0.143,
    transmon_control_bound = 0.153,
    cavity_control_bound   = 0.193,
    n_cavities=1, # TODO: add functionality for multiple cavities
    kwargs...
)
    @assert transmon_levels ∈ 1:3
    @assert n_cavities ∈ 1:2
    @assert length(ψ1) == length(ψf) == 2
    @assert ψ1[1] ∈ ['g', 'e'] && ψf[1] ∈ ['g', 'e']
    @assert parse(Int, ψ1[2]) ∈ 0:cavity_levels-2
    @assert parse(Int, ψf[2]) ∈ 0:cavity_levels-2

    params = Dict(
        :χ => χ,
        :κ => κ,
        :χGF => χGF,
        :α => α,
        :transmon_control_bound => transmon_control_bound,
        :cavity_control_bound => cavity_control_bound,
        :n_cavities => n_cavities,
    )

    if transmon_levels == 2

        transmon_g = [1, 0]
        transmon_e = [0, 1]

        H_drift = 2χ * kron(
            transmon_e * transmon_e',
            number(cavity_levels)
        ) + κ / 2 * kron(
            I(transmon_levels),
            quad(cavity_levels)
        )

        H_drive_transmon_R = kron(
            create(transmon_levels) + annihilate(transmon_levels),
            I(cavity_levels)
        )

        H_drive_transmon_I = kron(
            im * (create(transmon_levels) -
                annihilate(transmon_levels)),
            I(cavity_levels)
        )

        H_drive_cavity_R = kron(
            I(transmon_levels),
            create(cavity_levels) + annihilate(cavity_levels)
        )

        H_drive_cavity_I = kron(
            I(transmon_levels),
            im * (create(cavity_levels) -
                annihilate(cavity_levels))
        )

    elseif transmon_levels == 3

        transmon_g = [1, 0, 0]
        transmon_e = [0, 1, 0]
        transmon_f = [0, 0, 1]

        H_drift = α / 2 * kron(
            quad(transmon_levels),
            I(cavity_levels)
        ) + 2χ * kron(
            transmon_e * transmon_e',
            number(cavity_levels)
        ) + 2χGF * kron(
            transmon_f * transmon_f',
            number(cavity_levels)
        ) + κ / 2 * kron(
            I(transmon_levels),
            quad(cavity_levels)
        )

        println("howdy do!")

        H_drive_transmon_R = kron(
            create(transmon_levels) + annihilate(transmon_levels),
            I(cavity_levels)
        )

        H_drive_transmon_I = kron(
            1im * (annihilate(transmon_levels) - create(transmon_levels)),
            I(cavity_levels)
        )

        H_drive_cavity_R = kron(
            I(transmon_levels),
            create(cavity_levels) + annihilate(cavity_levels)
        )

        H_drive_cavity_I = kron(
            I(transmon_levels),
            1im * (annihilate(cavity_levels) - create(cavity_levels))
        )
    end

    ψinit = kron(
        ψ1[1] == 'g' ? transmon_g : transmon_e,
        cavity_state(parse(Int, ψ1[2]), cavity_levels)
    )

    ψgoal = kron(
        ψf[1] == 'g' ? transmon_g : transmon_e,
        cavity_state(parse(Int, ψf[2]), cavity_levels)
    )

    H_drives = [
        H_drive_transmon_R,
        H_drive_transmon_I,
        H_drive_cavity_R,
        H_drive_cavity_I
    ]

    a_bounds = [
        fill(transmon_control_bound, 2);
        fill(cavity_control_bound, 2 * n_cavities)
    ]

    return QuantumSystem(
        H_drift,
        H_drives,
        ψinit,
        ψgoal;
        a_bounds=a_bounds,
        params=params,
        kwargs...
    )
end


@doc raw"""
    MultiModeSystemNew(
        transmon_levels::Int,
        cavity_levels::Int;
        χ=2π * -0.5459e-3,
        κ=2π * 4e-6,
        χGF=2π * -1.01540302914e-3,
        α=-2π * 0.143,
        n_cavities=1
    )::QuantumSystemNew

Create a new `QuantumSystemNew` object for a transmon qubit with `transmon_levels` levels
coupled to a single cavity with `cavity_levels` levels.

The Hamiltonian for this system is given by

```math
H = \frac{\kappa}{2}\hat{a}^{\dagger}\hat{a}\left(\hat{a}^{\dagger}\hat{a}-1\right) +  2\ket{e}\bra{e}\chi\hat{a}^{\dagger}\hat{a} +  \left(\epsilon_{c}(t)+ \epsilon_{q}(t)+\mathrm{c.c.}\right)
```
"""
function MultiModeSystemNew(
    transmon_levels::Int,
    cavity_levels::Int;
    χ=2π * -0.5459e-3,
    κ=2π * 4e-6,
    χGF=2π * -1.01540302914e-3,
    α=-2π * 0.143,
    n_cavities=1 # TODO: add functionality for multiple cavities
)
    @assert transmon_levels ∈ 2:4
    @assert n_cavities ∈ 1:2
    @assert length(ψ1) == length(ψf) == 2

    params = Dict(
        :χ => χ,
        :κ => κ,
        :χGF => χGF,
        :α => α,
        :transmon_levels => transmon_levels,
        :cavity_levels => cavity_levels,
        :n_cavities => n_cavities,
    )

    if transmon_levels == 2

        transmon_g = [1, 0]
        transmon_e = [0, 1]

        H_drift =

            2χ * kron(
                transmon_e * transmon_e',
                number(cavity_levels)
            ) +

            κ / 2 * kron(
                I(transmon_levels),
                quad(cavity_levels)
            )

        H_drive_transmon_R = kron(
            create(transmon_levels) + annihilate(transmon_levels),
            I(cavity_levels)
        )

        H_drive_transmon_I = kron(
            im * (create(transmon_levels) -
                annihilate(transmon_levels)),
            I(cavity_levels)
        )

        H_drive_cavity_R = kron(
            I(transmon_levels),
            create(cavity_levels) + annihilate(cavity_levels)
        )

        H_drive_cavity_I = kron(
            I(transmon_levels),
            im * (create(cavity_levels) -
                annihilate(cavity_levels))
        )

    elseif transmon_levels == 3

        transmon_g = [1, 0, 0]
        transmon_e = [0, 1, 0]
        transmon_f = [0, 0, 1]

        H_drift = α / 2 * kron(
            quad(transmon_levels),
            I(cavity_levels)
        ) + 2χ * kron(
            transmon_e * transmon_e',
            number(cavity_levels)
        ) + 2χGF * kron(
            transmon_f * transmon_f',
            number(cavity_levels)
        ) + κ / 2 * kron(
            I(transmon_levels),
            quad(cavity_levels)
        )

        H_drive_transmon_R = kron(
            create(transmon_levels) + annihilate(transmon_levels),
            I(cavity_levels)
        )

        H_drive_transmon_I = kron(
            1im * (annihilate(transmon_levels) - create(transmon_levels)),
            I(cavity_levels)
        )

        H_drive_cavity_R = kron(
            I(transmon_levels),
            create(cavity_levels) + annihilate(cavity_levels)
        )

        H_drive_cavity_I = kron(
            I(transmon_levels),
            1im * (annihilate(cavity_levels) - create(cavity_levels))
        )
    end

    ψinit = kron(
        ψ1[1] == 'g' ? transmon_g : transmon_e,
        cavity_state(parse(Int, ψ1[2]), cavity_levels)
    )

    ψgoal = kron(
        ψf[1] == 'g' ? transmon_g : transmon_e,
        cavity_state(parse(Int, ψf[2]), cavity_levels)
    )

    H_drives = [
        H_drive_transmon_R,
        H_drive_transmon_I,
        H_drive_cavity_R,
        H_drive_cavity_I
    ]

    a_bounds = [
        fill(transmon_control_bound, 2);
        fill(cavity_control_bound, 2 * n_cavities)
    ]

    return QuantumSystem(
        H_drift,
        H_drives,
        ψinit,
        ψgoal;
        a_bounds=a_bounds,
        params=params,
        kwargs...
    )
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

        ψinit = vcat(transpose(hf["psi1"][:, :])...)
        ψgoal = vcat(transpose(hf["psif"][:, :])...)


        qubit_a_bounds = [0.018 * 2π, 0.018 * 2π]
        cavity_a_bounds = fill(0.03, length(H_drives) - 2)
        a_bounds = [qubit_a_bounds; cavity_a_bounds]

        system = QuantumSystem(
            H_drift,
            H_drives,
            ψinit,
            ψgoal,
            a_bounds,
            kwargs...
        )

        if return_data
            data = Dict()
            controls = copy(transpose(hf["controls"][:, :]))
            data["controls"] = controls
            Δt = hf["tlist"][2] - hf["tlist"][1]
            data["Δt"] = Δt
            return system, data
        else
            return system
        end
    end
end

end
