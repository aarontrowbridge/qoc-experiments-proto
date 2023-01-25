module PlottingILC

export animate_ILC
export animate_ILC_multimode

using ..IterativeLearningControl
using ..ILCExperiments
using ..ILCTrajectories

using CairoMakie

function animate_ILC(
    prob::ILCProblem,
    path::String;
    fps=5
)
    fig = Figure(resolution=(1200, 1200))


    Ygoalax = Axis(fig[1, :]; title="Y goal", xlabel=L"t")

    Ȳax = Axis(fig[2, :]; title="Ȳ", xlabel=L"t")

    Ygoal = hcat(prob.Ygoal.ys...)
    τs = prob.Ygoal.times

    series!(Ygoalax, τs, Ygoal;
        color=:seaborn_muted,
    )

    # axislegend(Ygoalax; position=:lb)

    Ȳ₁ = hcat(prob.Ȳs[1].ys...)

    Ȳsp = series!(Ȳax, τs, Ȳ₁;
        color=:seaborn_muted,
    )

    # axislegend(Ȳax; position=:lb)

    ΔYax = Axis(fig[3, :]; title="ΔY", xlabel=L"t")

    ΔY₁ = Ȳ₁ - Ygoal

    ΔYsp = series!(ΔYax, τs, ΔY₁;
        color=:seaborn_muted,
    )

    autolimits!(ΔYax)

    # axislegend(ΔYax; position=:lb)

    uax = Axis(fig[4, :]; title="a(t)", xlabel=L"t")

    U₁ = prob.Us[1]
    ts = prob.Ẑ.times

    Usp = series!(uax, ts, U₁;
        labels=["a_$j" for j = 1:prob.QP.dims.u]
    )

    record(fig, path, 1:length(prob.Ȳs); framerate=fps) do i
        Ȳ = hcat(prob.Ȳs[i].ys...)
        ΔY = Ȳ - Ygoal
        U = prob.Us[i]

        Ȳsp[2] = Ȳ
        ΔYsp[2] = ΔY
        Usp[2] = U
    end
end

function animate_ILC_multimode(
    prob::ILCProblem,
    path::String;
    fps=5
)
    fig = Figure(resolution=(1200, 1500))

    color = :glasbey_bw_minc_20_n256

    iter = Observable(0)

    Ygoal_ax = Axis(fig[1, :]; title=@lift("Y goal: iter = $($iter)"), xlabel=L"t")

    Ȳax = Axis(fig[2, :]; title=@lift("Ȳ: iter = $($iter)"), xlabel=L"t")

    Ygoal = hcat(prob.Ygoal.ys...)
    τs = prob.Ygoal.τs

    series!(Ygoal_ax, τs, Ygoal;
        color=color,
        markersize=5
    )

    # axislegend(Ygoalax; position=:lb)

    Ȳ₁ = hcat(prob.Ȳs[1].ys...)

    Ȳsp = series!(Ȳax, τs, Ȳ₁;
        color=color,
        markersize=5
    )

    # axislegend(Ȳax; position=:lb)

    ΔYax = Axis(fig[3, :]; title=@lift("ΔY: iter = $($iter)"), xlabel=L"t")

    ΔY₁ = Ȳ₁ - Ygoal

    ΔYsp = series!(ΔYax, τs, ΔY₁;
        color=color,
        markersize=5
    )

    autolimits!(ΔYax)

    # axislegend(ΔYax; position=:lb)

    U_ax = Axis(fig[4, :]; title=@lift("a(t): iter = $($iter)"), xlabel=L"t")

    U₁ = prob.Us[1]
    ts = prob.Ẑ.times

    U_sp = series!(U_ax, ts, U₁;
        labels=["a_$j" for j = 1:prob.QP.dims.u]
    )

    U_trans_ax = Axis(fig[5, :]; title=@lift("a(t) quantized: iter = $($iter)"), xlabel=L"t")

    U_trans_1 = prob.experiment.control_transform(prob.Us[1])

    U_trans_sp = series!(U_trans_ax, ts, U_trans_1;
        labels=["a_$j" for j = 1:prob.QP.dims.u]
    )

    # add axis plotting difference of previous 2 axes

    # ΔU_ax = Axis(fig[6, :]; title=@lift("Δa(t): iter = $($iter)"), xlabel=L"t")

    # ΔU_1 = U₁ - U_trans_1

    # ΔU_sp = series!(ΔU_ax, ts, ΔU_1;
    #     labels=["Δa_$j" for j = 1:prob.QP.dims.u]
    # )

    println(length(prob.Ȳs))
    record(fig, path, 1:length(prob.Ȳs); framerate=fps) do i
        iter[] = i

        Ȳ = hcat(prob.Ȳs[i].ys...)
        ΔY = Ȳ - Ygoal
        U = prob.Us[i]

        Ȳsp[2] = Ȳ
        ΔYsp[2] = ΔY
        U_sp[2] = U
        U_trans_sp[2] = prob.experiment.control_transform(U)
        # ΔU_sp[2] = U - prob.experiment.control_transform(U)
    end
end

function animate_ILC_multimode(
    prob::ILCProblemNew,
    path::String;
    fps=5
)
    fig = Figure(resolution=(1200, 1500))

    color = :glasbey_bw_minc_20_n256

    iter = Observable(0)

    Ygoal_ax = Axis(fig[1, :]; title=@lift("Y goal: iter = $($iter)"), xlabel=L"t")

    Ȳ_ax = Axis(fig[2, :]; title=@lift("Ȳ: iter = $($iter)"), xlabel=L"t")

    Ygoal = hcat(prob.Ygoal.ys...)
    τs = prob.Ygoal.τs

    series!(Ygoal_ax, τs, Ygoal;
        color=color,
        markersize=5
    )

    # axislegend(Ygoalax; position=:lb)

    Ȳ₁ = hcat(prob.Ȳs[1].ys...)

    Ȳ_sp = series!(Ȳ_ax, τs, Ȳ₁;
        color=color,
        markersize=5
    )

    # axislegend(Ȳax; position=:lb)

    ΔY_ax = Axis(fig[3, :]; title=@lift("ΔY: iter = $($iter)"), xlabel=L"t")

    ΔY₁ = Ȳ₁ - Ygoal

    ΔY_sp = series!(ΔY_ax, τs, ΔY₁;
        color=color,
        markersize=5
    )

    autolimits!(ΔY_ax)

    # axislegend(ΔYax; position=:lb)

    A_ax = Axis(fig[4, :]; title=@lift("a(t): iter = $($iter)"), xlabel=L"t")

    A₁ = prob.As[1]
    ts = times(prob.Ẑ)

    A_sp = series!(A_ax, ts, A₁;
        labels=["a_$j" for j = 1:prob.Ẑ.dims.a]
    )

    A_trans_ax = Axis(fig[5, :]; title=@lift("a(t) quantized: iter = $($iter)"), xlabel=L"t")

    A_trans_1 = prob.experiment.control_transform(A₁)

    A_trans_sp = series!(A_trans_ax, ts, A_trans_1;
        labels=["a_$j" for j = 1:prob.Ẑ.dims.a]
    )

    record(fig, path, 1:length(prob.Ȳs); framerate=fps) do i
        iter[] = i

        Ȳ = hcat(prob.Ȳs[i].ys...)
        ΔY = Ȳ - Ygoal
        A = prob.As[i]

        Ȳ_sp[2] = Ȳ
        ΔY_sp[2] = ΔY
        A_sp[2] = A
        A_trans_sp[2] = prob.experiment.control_transform(A)
    end
end



end
