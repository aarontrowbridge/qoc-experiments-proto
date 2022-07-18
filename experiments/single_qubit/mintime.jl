using QubitControl

σx = GATES[:X]
σy = GATES[:Y]
σz = GATES[:Z]

H_drift = σz / 2
H_drive = [σx / 2, σy / 2]

gate = Symbol(ARGS[1])

ψ0 = [1, 0]
ψ1 = [0, 1]

ψ = [ψ0, ψ1, (ψ0 + im * ψ1) / √2, (ψ0 - ψ1) / √2]

# ψ = [ψ0, ψ1, im * ψ0, im * ψ1]


system = SingleQubitSystem(H_drift, H_drive, gate, ψ)

T    = 1000
Δt   = 0.01
Q    = 0.0
Qf   = 500.0
R    = 0.001
loss = amplitude_loss
hess = false

options = Options(
    max_iter = 100,
    tol = 1e-6
)

iter = parse(Int, ARGS[2])

tol = 1e-8

plot_dir = "plots/single_qubit/min_time/two_controls"

for Rᵤ in [1e-7, 1e-10]

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
        Qf=Qf,
        R=R,
        Rᵤ=Rᵤ,
        Rₛ=Rₛ,
        eval_hessian=hess,
        loss=loss,
        options=options,
        min_time_options=min_time_options
    )

    plot_single_qubit(
        system,
        prob.subprob.trajectory,
        plot_path;
        fig_title="min time $gate gate on basis states (iter = $iter; tol = $tol)"
    )

    solve!(prob)

    plot_single_qubit(
        system,
        prob.subprob.trajectory,
        plot_path;
        fig_title="min time $gate gate on basis states (iter = $iter; tol = $tol)"
    )
end
