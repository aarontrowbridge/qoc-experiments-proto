# QubitControl.jl

This package directly implements quantum optimal control problems using MathOptInterface.jl and Ipopt.jl as the nonlinear solver backend.

## Problem formulation

Given a Hamiltonian for a qubit (or qubit coupled to harmonic oscillator(s)) system, of the form

$$
H(\mathbf{{a}}(t), t ) = H_{\text{drift}} + \sum_{i} a_i(t) H_{\text{drive}}^i
$$

we solve the optimization problem

$$
\begin{align}
\min_{\substack{\mathbf{x}\_1,\dots,\mathbf{x}\_T \\\ \mathbf{u}\_1, \dots, \mathbf{u}\_{T-1}}} \ \ \ \ 
& Q_T \cdot \ell(\tilde \psi_T^i, \tilde \psi_f^i) + \frac{1}{2} \sum_{t=1}^{T} R_t \cdot \mathbf{u}\_t^2\\ 
\text{subject to} \ \ \ \ 
& \mathbf{f}(\mathbf{x}\_{t+1}, \mathbf{u}\_{t+1}, \mathbf{x}\_t, \mathbf{u}\_t) = 0  \\
& \psi^i_1 = \ket{\psi^i}\_\text{init}, \ \psi^i_T = \ket{\psi^i}\_\text{goal} \\
& \smallint \mathbf{a}_1 = \mathbf{a}_1 = \mathrm{d}_t \mathbf{a}_1 = 0 \\  
& \smallint \mathbf{a}_T = \mathbf{a}_T = \mathrm{d}_t \mathbf{a}\_T = 0 \\
& |a^j_t| \leq a^j\_\text{bound} \\
\end{align}
$$

here the *state* vector $\mathbf{x}_t$ contains both the $n$ (`nqstates`) quantum isomorphism states $\tilde \psi^i_t$ (each of dimension `isodim`) and the augmented control states $\smallint \mathbf{a}_t$, $\mathbf{a}_t$, and $\mathrm{d}_t \mathbf{a}_t$ (the number of augmented state vector is `augdim`). The *action* vector $\mathbf{u}_t$ contains the second derivative of the *control* vector $\mathbf{a}_t$, which has dimension `ncontrols`. Thus, we have:

$$
\mathbf{x}_t = \begin{pmatrix} \tilde \psi^1_t \\\ \vdots \\\ \tilde \psi^n_t \\\ \smallint \mathbf{a}_t \\\ \mathbf{a}_t \\\ \mathrm{d}_t \mathbf{a}_t \end{pmatrix} \ \text{and} \ \ \ \mathbf{u}_t = \begin{pmatrix} \mathrm{d}^2_t \mathbf{a}_t  \end{pmatrix}
$$

So, $\dim(\mathbf{x}_t) =$ `nqstates * isodim + ncontrols * augdim`, and $\dim(\mathbf{u}_t)=$ `ncontrols`.

Finally, $\mathbf{f}(\mathbf{x}\_{t+1}, \mathbf{u}\_{t+1}, \mathbf{x}\_t, \mathbf{u}\_t)$, describes the dynamics of the all the variables in the system, where controls' dynamics are trivial and at a high level $\tilde \psi^i_t$ satisfies a discretized version of the isomorphic Schroedinger equation:

$$
{d \tilde \psi^i \over dt} = \widetilde{(- i H)}(\mathbf{a}(t), t) \tilde \psi^i
$$

I will the use the notation $G(H, \mathbf{a}(t))(t) = \widetilde{(- i H)}(\mathbf{a}(t), t)$, to describe this operator, which acts on the isomorphism wavefunctions 

$$
\tilde \psi = \begin{pmatrix} \psi^{\mathrm{Re}} \\ \psi^{\mathrm{Im}} \end{pmatrix}
$$ 

It can be shown then that

$$
G(H) = \begin{pmatrix} 1 & 0 \\\ 0 &1 \end{pmatrix} \otimes H^{\mathrm{Im}} - \begin{pmatrix} 0 & -1 \\\ 1 & 0 \end{pmatrix} \otimes H^{\mathrm{Re}}
$$

we then have the linear isomorphic dynamics equation:

$$
{d \tilde \psi \over dt} = G(\mathbf{a}(t),t) \tilde \psi
$$

where

$$
G(\mathbf{a}(t),t) = G(H_{\text{drift}}) + \sum_i a_i(t) G(H_{\text{drive}}^i) 
$$

## TODO: 

- [x] min time problem
- [ ] implement Hessian methods
- [x] document solver options (kinda completed, see `options.jl`)
- [ ] higher order Pade integrators
- [ ] exponential integrator
- [x] add constraints on a(t)
- [ ] write documentation 
- [ ] constraint types
- [ ] multimode system with specified control limits
- [ ] implicit \ddot a(t)
- [ ] add ability to change linear solver
  - [ ] mac
  - [ ] ubuntu
  - [ ] arch (manjaro)
