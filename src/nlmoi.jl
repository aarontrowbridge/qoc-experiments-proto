module NonLinearMOI

using ..Problems
using ..Dynamics

import MathOptInterface as MOI
import Ipopt

function solve(prob::MultiQubitProblem{N}) where N
    model = Ipopt.Optimizer()




end

end
