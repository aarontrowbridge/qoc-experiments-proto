module Utils

export slice
export index

index(t::Int, pos::Int, dim::Int; start=0) = start + (t - 1) * dim + pos
index(t, dim) = index(t, dim, dim)

slice(t, pos1, pos2, dim) = index(t, pos1, dim):index(t, pos2, dim)
slice(t, pos, dim) = slice(t, 1, pos, dim)
slice(t, dim; stretch=0) = slice(t, 1, dim + stretch, dim)

end
