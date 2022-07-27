module Utils

export slice
export index
export generate_file_path

index(t::Int, pos::Int, dim::Int) = (t - 1) * dim + pos

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
    max_numeric_prefix = -1
    for (_, _, files) in walkdir(path)
        for file_name_ in files
            if occursin("_$(file_name).$(extension)", file_name_)
                numeric_prefix = parse(Int, split(file_name_, "_")[1])
                max_numeric_prefix = max(numeric_prefix, max_numeric_prefix)
            end
        end
    end

    file_path = joinpath(path, "$(lpad(max_numeric_prefix + 1, 5, '0'))_$(file_name).$(extension)")
    return file_path
end

end
