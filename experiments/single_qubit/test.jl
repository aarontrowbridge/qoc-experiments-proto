using QubitControl

σx = GATES[:X]
σz = GATES[:Z]

f = 0.5

H_drift = f * σz / 2
H_drive = σx / 2

gate = :X

ψ1 = [1, 0]

prob = SingleQubitProblem(H_drift, H_drive, gate, ψ1; T=10)

solve(prob)
