using Zygote
using LinearAlgebra

d = rand(3)

D = Diagonal(d)

fieldnames(typeof(D))

D.diag == diag(D)

LinearAlgebra.diag(D::Diagonal) = D.diag

function exp_diag_function(D::Diagonal)
    return Diagonal(exp.(diag(D)))
end

function exp_diag_field_name(D::Diagonal)
    return Diagonal(exp.(D.diag))
end

∇_diag_function(x) = Zygote.gradient(x -> tr(exp_diag_function(x * D)), x)

∇_diag_function(1.0)

∇_diag_field_name(x) = Zygote.gradient(x -> tr(exp_diag_field_name(x * D)), x)

∇_diag_field_name(1.0)

∇_built_in(x) = Zygote.gradient(x -> tr(exp(x * D)), x)

∇_built_in(1.0)
