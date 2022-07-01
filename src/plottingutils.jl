module PlottingUtils

export plot_single_qubit_1_qstate_with_controls
export plot_single_qubit_2_qstate_with_controls
export plot_single_qubit_2_qstate_with_seperated_controls

using ..Utils
using ..Trajectories
using ..Problems

using LaTeXStrings
using CairoMakie

function plot_single_qubit_1_qstate_with_controls(
    prob::QubitProblem,
    filename::String;
    kwargs...
)
    return plot_single_qubit_1_qstate_with_controls(
        prob.trajectory,
        filename,
        prob.system.isodim,
        prob.system.control_order,
        prob.T;
        kwargs...
    )
end

function plot_single_qubit_1_qstate_with_controls(
    traj::TrajectoryData,
    filename::String,
    isodim::Int,
    control_order::Int,
    T::Int;
    fig_title=nothing,
    i=1
)
    xs = traj.states
    us = traj.controls
    ts = traj.times

    ψ̃s = [xs[t][slice(i, isodim)] for t = 1:T]
    ψ̃s = hcat(ψ̃s...)

    as = [[xs[t][(end - control_order):end]; us[t]] for t = 1:T]
    as = hcat(as...)

    fig = Figure(resolution=(1200, 1000))

    ax1 = Axis(fig[1, 1]; title="qubit components", xlabel=L"t")
    ax2 = Axis(fig[1, 2]; title="control", xlabel=L"t")
    ax3 = Axis(fig[2, 1]; title="first derivative of control", xlabel=L"t")
    ax4 = Axis(fig[2, 2]; title="second derivative of control", xlabel=L"t")

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
    kwargs...
)
    return plot_single_qubit_2_qstate_with_controls(
        prob.trajectory,
        filename,
        prob.system.isodim,
        prob.system.control_order,
        prob.T;
        kwargs...
    )
end

function plot_single_qubit_2_qstate_with_controls(
    traj::TrajectoryData,
    filename::String,
    isodim::Int,
    control_order::Int,
    T::Int;
    fig_title=nothing
)

    xs = traj.states
    us = traj.controls
    ts = traj.times

    ψ̃¹s = [xs[t][slice(1, isodim)] for t = 1:T]
    ψ̃¹s = hcat(ψ̃¹s...)

    ψ̃²s = [xs[t][slice(2, isodim)] for t = 1:T]
    ψ̃²s = hcat(ψ̃²s...)

    as = [[xs[t][(end - control_order + 1):end]; us[t]] for t = 1:T]
    as = hcat(as...)
    as[end, end] = as[end, end-1]

    fig = Figure(resolution=(1200, 1000))

    ax_ψ̃¹ = Axis(fig[1, 1]; title="qubit components: U(t)|0⟩", xlabel=L"t")
    ax_ψ̃² = Axis(fig[1, 2]; title="qubit components: U(t)|1⟩", xlabel=L"t")
    ax_as = Axis(fig[2, :]; title="controls", xlabel=L"t")

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

function plot_single_qubit_2_qstate_with_seperated_controls(
    traj::TrajectoryData,
    filename::String,
    isodim::Int,
    control_order::Int,
    T::Int;
    fig_title=nothing
)

    xs = traj.states
    us = traj.controls
    ts = traj.times

    ψ̃¹s = [xs[t][slice(1, isodim)] for t = 1:T]
    ψ̃¹s = hcat(ψ̃¹s...)

    ψ̃²s = [xs[t][slice(2, isodim)] for t = 1:T]
    ψ̃²s = hcat(ψ̃²s...)

    as = [[xs[t][(end - control_order + 1):end]; us[t]] for t = 1:T]
    as = hcat(as...)
    as[end, end] = as[end, end-1]

    fig = Figure(resolution=(1200, 1000))

    ax_ψ̃¹ = Axis(
        fig[1:3, 1];
        title="qubit components: U(t)|0⟩",
        xlabel=L"t"
    )

    ax_ψ̃² = Axis(
        fig[1:3, 2];
        title="qubit components: U(t)|1⟩",
        xlabel=L"t"
    )


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


    for i = 0:control_order
        ax = Axis(
            fig[4 + i, :];
            xlabel = L"t"
        )

        lines!(
            ax,
            ts,
            as[1 + i, :];
            label = i == 0 ?
                L"a(t)" :
                latexstring(
                    "\\mathrm{d}^{",
                    i == 1 ? "" : "$i",
                    "}_t a"
                )
                # L"\mathrm{d}^{"*(i == 1 ? "" : "$i")*L"}_t a"
                # L"\mathrm{d}^{ \$\(i\) }_t a"
        )

        axislegend(ax; position=:rt)
    end

    if !isnothing(fig_title)
        Label(fig[0,:], fig_title; textsize=30)
    end

    save(filename, fig)
end


end
