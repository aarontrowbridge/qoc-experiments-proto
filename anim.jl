using CairoMakie

x = 0:0.1:10
ys(t) = exp(t) .* hcat(sin.(t .+ x), cos.(t .+ x))'

fig = Figure(resolution=(600, 400))

ax = Axis(fig[1, 1], xlabel="x", ylabel="y", title="sin(x) and cos(x)")

sp = series!(ax, x, ys(0))

record(fig, "test.gif", 1:10; framerate=5) do t
    sp[2] = ys(t)
    autolimits!(ax)
end
