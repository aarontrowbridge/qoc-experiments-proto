module Utils

export slice

slice(t, dim; stretch=0) = (1 + (t - 1)*dim):(t*dim + stretch)

end
