WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "experiments", "ilqoc", "measurement.jl"))



const EXPERIMENT_NAME = "8-5-2022-transmon_no_int_a"
plot_path = generate_file_path("png", EXPERIMENT_NAME * "_iter_$(iter)", "plots/")

#system parameters

ω = 2π * 4.96 #GHz
α = -2π * 0.143 #GHz
levels = 3

ψg = [1. + 0*im, 0 , 0]
ψe = [0, 1. + 0*im, 0]

ψ1 = [ψg, ψe]
ψf = [-im*ψe, -im*ψg]

H_drift = α/2 * quad(levels)
H_drive = [create(levels) + annihilate(levels),
1im * (create(levels) - annihilate(levels))]

system = QuantumSystem(
    H_drift, 
    H_drive,
    ψ1 = ψ1,
    ψf = ψf,
    control_bounds = [2π * 19e-3,  2π * 19e-3]
)