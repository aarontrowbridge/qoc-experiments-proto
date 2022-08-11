# QubitControl.jl

This package implements direct collocation to solve quantum optimal control problems. It currently relies on MathOptInterface.jl, with Ipopt.jl as the nonlinear solver backend.

## Installation

To install and use this repo on your own machine:

1. clone the repo into a directory of your choosing
2. open a julia REPL from the repo directory and enter the package manager via `julia> ]`
3. run `(@v1.7) pkg> activate .` 
4. then run `(QubitControl) pkg> instantiate`
5. now the package can be used via `using QubitControl`


## Problem formulation

Given a Hamiltonian for a qubit (or qubit coupled to harmonic oscillator(s)) system of the form

$$
H(\mathbf{{a}}(t), t ) = H_{\text{drift}} + \sum_{j} a^j(t) H_{\text{drive}}^j
$$

we solve the optimization problem

$$
\begin{align}
\underset{\mathbf{x}\_{1:T}, \ \mathbf{u}\_{1:T-1}}{\text{minimize}} \quad
& \sum_{i=1}^n Q_T \cdot \ell(\tilde \psi_T^i, \tilde \psi_\text{goal}^i) + \frac{1}{2} \sum_{t=1}^{T-1} R_t \cdot \mathbf{u}\_t^2\\ 
\text{subject to} \quad 
& \mathbf{f}(\mathbf{x}\_{t+1}, \mathbf{x}\_t, \mathbf{u}\_t) = \mathbf{0}  \\
&  \tilde \psi^i_1 = \tilde \psi^i\_\text{init} \\
& \tilde \psi^1_T = \tilde \psi^1\_\text{goal} \quad \text{if }\ \small \textsf{pin\\_first\\_qstate = true}\\
& \smallint \mathbf{a}_1 = \mathbf{a}_1 = \mathrm{d}_t \mathbf{a}_1 = \mathbf{0} \\  
& \smallint \mathbf{a}_T = \mathbf{a}_T = \mathrm{d}_t \mathbf{a}\_T = \mathbf{0} \\
& |a^j_t| \leq a^j\_\text{bound} \\
\end{align}
$$

The *state* vector $\mathbf{x}_t$ contains both the $n$ (`nqstates`) quantum isomorphism states $\tilde \psi^i_t$ (each of dimension `isodim = 2*ketdim`) and the augmented control states $\smallint \mathbf{a}_t$, $\mathbf{a}_t$, and $\mathrm{d}_t \mathbf{a}_t$ (the number of augmented state vector is `augdim`). The *action* vector $\mathbf{u}_t$ contains the second derivative of the *control* vector $\mathbf{a}_t$, which has dimension `ncontrols`. Thus, we have:

$$
\mathbf{x}_t = \begin{pmatrix} \tilde \psi^1_t \\\ \vdots \\\ \tilde \psi^n_t \\\ \smallint \mathbf{a}_t \\\ \mathbf{a}_t \\\ \mathrm{d}_t \mathbf{a}_t \end{pmatrix} \ \text{and} \ \ \ \mathbf{u}_t = \begin{pmatrix} \mathrm{d}^2_t \mathbf{a}_t  \end{pmatrix}
$$

So, $\dim(\mathbf{x}_t) =$ `nqstates * isodim + ncontrols * augdim = nstates`, and $\dim(\mathbf{u}_t)=$ `ncontrols`.

### cost functions

Currently the code is set up to support any quantum state cost function $\ell$; the default choice is

$$
\ell(\tilde\psi, \tilde\psi\_{goal}) = 1 - |\braket{\psi | \psi\_{\text{goal}}}|^2
$$

### dynamics

Finally, $\mathbf{f}(\mathbf{x}\_{t+1}, \mathbf{x}\_t, \mathbf{u}\_t)$ describes the dynamics of all the variables in the system, where the controls' dynamics are trivial and formally $\tilde \psi^i_t$ satisfies a discretized version of the isomorphic Schroedinger equation:

$$
{d \tilde \psi^i \over dt} = \widetilde{(- i H)}(\mathbf{a}(t), t) \tilde \psi^i
$$

I will the use the notation $G(H)(\mathbf{a}(t), t) = \widetilde{(- i H)}(\mathbf{a}(t), t)$, to describe this operator (the Generator of time translation), which acts on the isomorphic quantum state vectors 

$$
\tilde \psi = \begin{pmatrix} \psi^{\mathrm{Re}} \\\ \psi^{\mathrm{Im}} \end{pmatrix}
$$ 

It can be shown that

$$
G(H) =  - \begin{pmatrix} 0 & -1 \\\ 1 & 0 \end{pmatrix} \otimes H^{\mathrm{Re}} + \begin{pmatrix} 1 & 0 \\\ 0 &1 \end{pmatrix} \otimes H^{\mathrm{Im}}
$$

where $\otimes$ is the Kronecker product.  We then have the linear isomorphism dynamics equation:

$$
{d \tilde \psi \over dt} = G(\mathbf{a}(t),t) \tilde \psi
$$

where

$$
G(\mathbf{a}(t),t) = G(H_{\text{drift}}) + \sum_j a^j(t) G(H_{\text{drive}}^j) 
$$

The implicit dynamics constraint function $\mathbf{f}$ can be decomposed as follows:

$$
\mathbf{f}(\mathbf{x}\_{t+1}, \mathbf{x}\_t, \mathbf{u}_t) 
= \begin{pmatrix} 
  \mathbf{P}^m (\tilde \psi^1\_{t+1}, \tilde \psi^1\_t, \mathbf{a}\_t) \\\ 
  \vdots \\\
  \mathbf{P}^m (\tilde \psi^n\_{t+1}, \tilde \psi^n\_t, \mathbf{a}\_t) \\\
  \smallint \mathbf{a}\_{t+1} - \left(\smallint \mathbf{a}\_t + \mathbf{a}\_t \cdot \Delta t\_t  \right) \\\
  \mathbf{a}\_{t+1} - \left(\mathbf{a}\_t + \mathrm{d}\_t \mathbf{a}\_t \cdot \Delta t\_t  \right) \\\
  \mathrm{d}\_t \mathbf{a}\_{t+1} - \left(\mathrm{d}\_t \mathbf{a}\_t + \mathbf{u}\_t \cdot \Delta t\_t \right)
  \end{pmatrix}
$$

## TODO: 

- [ ] min time problem reimplementation
  - [x] implementation
  - [x] tests
  - [ ] add details to notes
- [ ] analytic derivatives
  - [ ] integrators
    - [x] 2nd order Pade
    - [x] 4th order Pade
    - [ ] exponential
  - [x] objective
- [x] document solver options (kinda completed, see `options.jl`)
- [x] integrator functor type
  - [x] higher order Pade integrators
  - [x] exponential integrator
- [x] add constraints on a(t)
- [ ] write documentation 
- [x] constraint types
- [x] systems
  - [x] multimode system with specified control limits
  - [x] two qubit system (try CNOT)
  - [x] transmon system
  - [x] multimode system
- [ ] implicit $\ddot a(t)$
- [ ] add ability to change linear solver
  - [ ] mac
  - [ ] ubuntu
  - [ ] arch (manjaro)
- [ ] add ability to save and load trajectories
