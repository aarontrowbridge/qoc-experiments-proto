module ProblemSolvers

export solve!

using ..Constraints
using ..Problems
# using ..MinTimeProblems
using ..ProblemUtils

using MathOptInterface
const MOI = MathOptInterface

function solve!(
    prob::QuantumControlProblem;
    init_traj=prob.trajectory,
    save_path=nothing,
    controls_save_path=nothing,
)
    initialize_trajectory!(prob, init_traj)

    MOI.optimize!(prob.optimizer)

    update_traj_data!(prob)

    if ! isnothing(save_path)
        save_problem(prob, save_path)
    end

    if ! isnothing(controls_save_path)
        # TODO: implement this @aditya
        save_controls(prob, controls_save_path)
    end
end

# function solve!(
#     prob::MinTimeQuantumControlProblem;
#     save_path=nothing,
#     solve_subprob=true,
# )
#     if solve_subprob
#         if prob.subprob isa FixedTimeProblem
#             solve!(prob.subprob)
#         else
#             println()
#             @info "Subproblem is not a FixedTimeProblem. Skipping solve."
#             println()
#         end
#     end

#     init_traj = prob.subprob.trajectory

#     initialize_trajectory!(prob, init_traj)

#     n_wfn_states = prob.subprob.system.n_wfn_states

#     # constrain endpoints to match subprob solution

#     if prob.subprob.params[:pin_first_qstate]
#         isodim = prob.subprob.system.isodim
#         ψ̃T_con! = EqualityConstraint(
#             prob.subprob.trajectory.T,
#             (isodim + 1):n_wfn_states,
#             init_traj.states[end][(isodim + 1):n_wfn_states],
#             prob.subprob.system.vardim;
#             name="final qstate constraint"
#         )
#     else
#         ψ̃T_con! = EqualityConstraint(
#             prob.subprob.trajectory.T,
#             1:n_wfn_states,
#             init_traj.states[end][1:n_wfn_states],
#             prob.subprob.system.vardim;
#             name="final qstate constraint"
#         )
#     end

#     ψ̃T_con!(prob.optimizer, prob.variables)

#     MOI.optimize!(prob.optimizer)

#     update_traj_data!(prob)

#     if ! isnothing(save_path)
#         save_problem(prob, save_path)
#     end
# end


# TODO: add functionality to vizualize Δt distribution




end
