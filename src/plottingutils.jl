module PlottingUtils

export plot_single_qubit_1_qstate_with_controls
export plot_single_qubit_2_qstate_with_controls

using ..Utils
using ..Problems

using CairoMakie

function plot_single_qubit_1_qstate_with_controls(
    prob::QubitProblem,
    filename::String;
    fig_title=nothing,
    i=1
)
    xs = prob.trajectory.states
    us = prob.trajectory.actions

    ψ̃s = [xs[t][slice(i, prob.system.isodim)] for t = 1:prob.T]
    ψ̃s = hcat(ψ̃s...)

    as = [[xs[t][(end - prob.system.control_order):end]; us[t]] for t = 1:prob.T]
    as = hcat(as...)

    fig = Figure(resolution=(1200, 1000))

    ax1 = Axis(fig[1, 1]; title="qubit components", xlabel=L"t")
    ax2 = Axis(fig[1, 2]; title="control", xlabel=L"t")
    ax3 = Axis(fig[2, 1]; title="first derivative of control", xlabel=L"t")
    ax4 = Axis(fig[2, 2]; title="second derivative of control", xlabel=L"t")

    ts = (1:prob.T) .* prob.Δt

    series!(ax1, ts, ψ̃s;
        labels=[
            L"\mathrm{Re} (\psi_0)",
            L"\mathrm{Re} (\psi_1)",
            L"\mathrm{Im} (\psi_0)",
            L"\mathrm{Im} (\psi_1)"
        ]
    )
    axislegend(ax1; position=:cb)

    lines!(ax2, ts, as[2,:]; label=L"a(t)")
    axislegend(ax2; position=:cb)

    lines!(ax3, ts, as[3,:]; label=L"\mathrm{d}_t a")
    axislegend(ax3; position=:cb)

    lines!(ax4, ts[1:end-1], as[4,1:end-1]; label=L"\mathrm{d}^2_t (t)")
    axislegend(ax4; position=:cb)

    if !isnothing(fig_title)
        Label(fig[0,:], fig_title; textsize=30)
    end

    save(filename, fig)
end

function plot_single_qubit_2_qstate_with_controls(
    prob::QubitProblem,
    filename::String;
    fig_title=nothing
)
    xs = prob.trajectory.states
    us = prob.trajectory.actions

    ψ̃¹s = [xs[t][slice(1, prob.system.isodim)] for t = 1:prob.T]
    ψ̃¹s = hcat(ψ̃¹s...)

    ψ̃²s = [xs[t][slice(2, prob.system.isodim)] for t = 1:prob.T]
    ψ̃²s = hcat(ψ̃²s...)

    as = [[xs[t][(end - prob.system.control_order + 1):end]; us[t]] for t = 1:prob.T]
    as = hcat(as...)
    as[end, end] = as[end, end-1]

    fig = Figure(resolution=(1200, 1000))

    ax_ψ̃¹ = Axis(fig[1, 1]; title="qubit components: U(t)|0⟩", xlabel=L"t")
    ax_ψ̃² = Axis(fig[1, 2]; title="qubit components: U(t)|1⟩", xlabel=L"t")
    ax_as = Axis(fig[2, :]; title="controls", xlabel=L"t")

    ts = (1:prob.T) .* prob.Δt

    series!(ax_ψ̃¹, ts, ψ̃¹s;
        labels=[
            L"\mathrm{Re} (\psi^1_0)",
            L"\mathrm{Re} (\psi^1_1)",
            L"\mathrm{Im} (\psi^1_0)",
            L"\mathrm{Im} (\psi^1_1)"
        ]
    )
    axislegend(ax_ψ̃¹; position=:lb)

    series!(ax_ψ̃², ts, ψ̃²s;
        labels=[
            L"\mathrm{Re} (\psi^2_0)",
            L"\mathrm{Re} (\psi^2_1)",
            L"\mathrm{Im} (\psi^2_0)",
            L"\mathrm{Im} (\psi^2_1)"
        ]
    )
    axislegend(ax_ψ̃²; position=:lb)

    series!(ax_as, ts, as;
        labels=[
            L"a(t)",
            L"\mathrm{d}_t a",
            L"\mathrm{d}^2_t a"
        ]
    )
    axislegend(ax_as; position=:rt)

    if !isnothing(fig_title)
        Label(fig[0,:], fig_title; textsize=30)
    end

    save(filename, fig)
end


end
