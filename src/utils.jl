module Utils

export index
export slice

"""
this module contains helper functions for indexing and taking slices of the full problem variable vector

definitions:

the problem vector: Z = [z₁, z₂, ..., zT]

    knot point:
        zₜ = [xₜ, uₜ]

    augmented state vector:
        xₜ = [ψ̃ₜ, ψ̃²ₜ, ..., ψ̃ⁿₜ, ∫aₜ, aₜ, daₜ, ..., dᶜ⁻¹aₜ]

    where:
        c = control_order


also, below, we use dim(zₜ) = dim

examples:

Z[index(t, pos, dim)] = zₜ[pos]
Z[index(t, dim)]      = zₜ[dim]

Z[slice(t, pos1, pos2, dim)]      = zₜ[pos1:pos2]
Z[slice(t, pos, dim)]             = zₜ[1:pos]
Z[slice(t, dim)]                  = zₜ[1:dim]
Z[slice(t, dim; stretch=stretch)] = zₜ[1:(dim + stretch)]
Z[slice(t, indices, dim)]         = zₜ[indices]

the functions are also used to access the zₜ vectors, e.g.

zₜ[slice(i, isodim)]                             = ψ̃ⁱₜ
zₜ[n_wfn_states .+ slice(1, ncontrols)]          = ∫aₜ
zₜ[n_wfn_states .+ slice(2, ncontrols)]          = aₜ
zₜ[n_wfn_states .+ slice(augdim + 1, ncontrols)] = uₜ = ddaₜ
"""

index(t::Int, pos::Int, dim::Int) = dim * (t - 1) + pos

index(t, dim) = index(t, dim, dim)


slice(t, pos1, pos2, dim) =
    index(t, pos1, dim):index(t, pos2, dim)

slice(t, pos, dim) = slice(t, 1, pos, dim)

slice(t, dim; stretch=0) = slice(t, 1, dim + stretch, dim)

function generate_file_path(extension, file_name, path)
    # Ensure the path exists.
    mkpath(path)

    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory.
    max_numeric_suffix = -1
    for (_, _, files) in walkdir(path)
        for file_name_ in files
            if occursin("_$(file_name).$(extension)", file_name_)

                numeric_suffix = parse(
                    Int,
                    split(file_name_, "_")[end]
                )

                max_numeric_suffix = max(
                    numeric_suffix,
                    max_numeric_suffix
                )
            end
        end
    end

    file_path = joinpath(
        path,
        file_name *
        "_$(lpad(max_numeric_suffix + 1, 5, '0')).$(extension)"
    )

    return file_path
end

slice(t::Int, indices::AbstractVector{Int}, dim::Int) =
    dim * (t - 1) .+ indices

end
