using QubitControl

σx = GATES[:X]
σy = GATES[:Y]
σz = GATES[:Z]

H_drift = σz / 2
H_drives = [σx / 2, σy / 2]

gate = Symbol(ARGS[1])

ψ0 = [1, 0]
ψ1 = [0, 1]

ψ = [ψ0, ψ1, (ψ0 + im * ψ1) / √2, (ψ0 - ψ1) / √2]

system = SingleQubitSystem(H_drift, H_drives, gate, ψ)

T  = parse(Int, ARGS[2])
Δt = 0.01
Q  = 500.0
R  = 0.001

loss = infidelity_loss

a_bound = [1.0, 0.5]

options = Options(
    max_iter = parse(Int, ARGS[end-1]),
    tol = 1e-6
)

iter = parse(Int, ARGS[end])

tol = 1e-5

plot_dir = "plots/single_qubit/min_time/two_controls"

for Rᵤ in [1e-3, 1e-5]

    Rₛ = Rᵤ

    plot_file = "$(gate)_gate_Ru_$(Rᵤ)_Rs_$(Rₛ)_tol_$(tol)_iter_$(iter)_pinned.png"

    plot_path = joinpath(plot_dir, plot_file)

    min_time_options = Options(
        max_iter = iter,
        tol = tol
    )

    prob = MinTimeProblem(
        system,
        T;
        Δt=Δt,
        Q=Q,
        R=R,
        Rᵤ=Rᵤ,
        Rₛ=Rₛ,
        a_bound=a_bound,
        loss=loss,
        options=options,
        min_time_options=min_time_options
    )

    plot_single_qubit(
        system,
        prob.subprob.trajectory,
        plot_path
    )

    solve!(prob)

    plot_single_qubit(
        system,
        prob.subprob.trajectory,
        plot_path
    )
end
