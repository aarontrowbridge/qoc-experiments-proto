using PicoOld
using Test

using ForwardDiff
using FiniteDiff
using SparseArrays
using LinearAlgebra

const ATOL = 1e-4

include("test_utils.jl")

@testset "testing Pico.jl" begin
    include("objective_tests.jl")
    include("dynamics_tests.jl")
    include("problem_tests.jl")
end
