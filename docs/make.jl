using Documenter
using Pico

push!(LOAD_PATH, "../src/")

makedocs(
    sitename = "Pico.jl",
    format=Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
)

deploydocs(
    repo="github.com/aarontrowbridge/Pico.jl.git",
)
