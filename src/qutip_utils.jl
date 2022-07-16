__precompile__()
module QuTiPUtils

export get_qutip_matrix
export py_sparse_matrix_to_julia
export load_qutip_object

using PyCall
using SparseArrays

const qt = PyNULL()
const sp = PyNULL()

function __init__()
    copy!(qt, pyimport_conda("qutip", "qutip"))
    copy!(sp, pyimport_conda("scipy.sparse", "scipy"))
end

function load_qutip_object(path::String)
    Qobj = qt.fileio.qload(path)
    return Qobj
end

function py_sparse_matrix_to_julia(A)
    A = sp.csc_matrix(A)
    m, n = A.shape
    colptr = A.indptr .+ 1
    rowval = A.indices .+ 1
    nzval = A.data
    return SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

function get_qutip_matrix(path::String; to_static=true)
    Qobj = qt.fileio.qload(path)
    py_matrix = Qobj.data
    return py_sparse_matrix_to_julia(py_matrix)
end

end
