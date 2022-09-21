# Systems

Pico builds discretized optimization *problems* from *systems*, which define the system-specific parameters; e.g. for a `QuantumSystem`, the required information includes the Hamiltonian terms, initial and goal states, and bounds on the control variables.  All systems are subtypes of the abstract type `AbstractSystem`.

```@docs
AbstractSystem
```

## Quantum systems

The primary focus of Pico is quantum optimal control and currently the only defined system type is the `QuantumSystem`. (Although, the core methods in this package are applicable to any linear dynamical system).

```@docs
QuantumSystem
```


`QuantumSystem`s can be constructed in multiple ways. The simplest is to provide the Hamiltonian terms, initial and goal states, and bounds on the control variables. The following constructor provides this functionality:

```@docs
QuantumSystem(
    ::Matrix,
    ::Union{Matrix{T}, Vector{Matrix{T}}},
    ::Union{Vector{C1}, Vector{Vector{C1}}},
    ::Union{Vector{C2}, Vector{Vector{C2}}},
    ::Vector{Float64};
) where {C1 <: Number, C2 <: Number, T <: Number}
```

TODO: include example here

### Single-qubit systems

TODO: build this constructor

### Two-qubit systems

TODO: build this constructor

### Multimode systems

A convenience function function for building a `QuantumSystem` composed of a single transmon qubit coupled to an array of cavity resonators is provided:

```@docs
MultiModeSystem
```