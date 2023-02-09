"""
    problem tests
"""

@testset "problem tests" begin



σx = GATES[:X]
σy = GATES[:Y]
σz = GATES[:Z]

H_drift = σz / 2
H_drive = [σx / 2, σy / 2]

gate = :X

ψ0 = [1, 0]
ψ1 = [0, 1]

# ψ = [ψ0, ψ1, (ψ0 + im * ψ1) / √2, (ψ0 - ψ1) / √2]
ψ = [ψ0, ψ1]
ψf = apply.(gate, ψ)

a_bounds = [1.0, 0.5]

system = QuantumSystem(
    H_drift,
    H_drive,
    ψ,
    ψf;
    a_bounds=a_bounds,
)

@testset "testing saving and loading of problems" begin

    fixed_time_save_path = generate_file_path(
        "jld2",
        "test_save",
        pwd()
    )

    prob = QuantumControlProblem(system)

    save_problem(prob, fixed_time_save_path)

    @testset "fixed time problems" begin
        prob_loaded = load_problem(fixed_time_save_path)
        data_loaded = load_data(fixed_time_save_path)

        @test prob_loaded isa QuantumControlProblem
        @test data_loaded isa ProblemData
    end

    rm(fixed_time_save_path)
end


end
