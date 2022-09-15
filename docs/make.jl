using Documenter
using Pico

push!(LOAD_PATH, "../src/")

makedocs(
    sitename = "Pico.jl",
    format=Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    pages=[
        "Introduction" => "intro.md",
        "Getting Started" => "getting_started.md",
        "Manual" => [
            "Systems" => "systems.md",
            "Objectives" => "objectives.md",
            "Costs" => "costs.md",
            "Constraints" => "constraints.md",
            "Integrators" => "integrators.md",
            "Problems" => "problems.md",
            "Iterative Learning Control" => "ilc.md",
            "Trajectories" => "trajectories.md",
            "Plotting" => "plotting.md",
        ],
        "Examples" => "examples.md",
        "API" => "api.md",
    ],
)

deploydocs(
    repo="github.com/aarontrowbridge/Pico.jl.git",
)
