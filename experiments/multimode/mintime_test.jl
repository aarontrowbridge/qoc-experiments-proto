using PicoOld
using Test

T = 10

P = :FourthOrderPade

sys = MultiModeSystem(3, 14, "g0", "g1")

N = sys.vardim*T
Z_indices = 1:N
Δt_indices = N .+ (1:T-1)

D = MinTimeQuantumDynamics(sys, P, Z_indices, Δt_indices, T)

Z = randn(N)

μ = randn(sys.nstates * (T - 1))

Δt = rand(T-1)

Z̄ = [Z; Δt]

D.F(Z̄)

D.∂F(Z̄)

D.μ∂²F(Z̄, μ)
