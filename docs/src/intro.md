# Introduction

**Pico.jl** is a multifaceted toolbox for formulating and solving quantum optimal control problems.  Pico implements direct collocation using novel Pade integrators; the name is derived from *Pade integrator collocation*.

## Formal problem statement 

If we want to find a drive pulse ``\vb{a}(t)`` that maps

```math
\ket{\psi^i_{\text{init}}} \to \ket{\psi^i_{\text{goal}}} \quad \forall i
```
using a time-dependent Hamiltonian of the form

```math
H(\vb{a}(t)) = H_{\text{drift}} + \sum_j a^j(t) H_{\text{drive}}^j,
```

formally, we seek to find the solution to the optimization problem

```math
\begin{align*}
\underset{\ket{\psi^{1:n}_{1:T}},\ \vb{a}_{1:T-1}}{\text{minimize}} & \quad Q \cdot \sum_i \qty(1 - \abs{\braket{\psi^i_T}{\psi^i_{\text{goal}}}}^2) + \sum_t \vb{a}_t^\top R_t \vb{a}_t \\

\text{subject to} & \quad \ket{\psi^i_{t+1}} = e^{-i \Delta t H(\vb{a}_t)} \ket{\psi^i_t} \quad \forall i,t \\
& \quad \ket{\psi^i_1} = \ket{\psi^i_{\text{init}}} \quad \forall i \\
& \quad \ket{\psi^i_T} = \ket{\psi^i_{\text{goal}}} \quad \forall i \\
& \quad \abs{a^j_t} \leq A^j \quad \forall j,t
\end{align*}
```

## A well-behaved, isomorphic problem reformulation 
