module Systems

using LinearAlgebra

using OperatorsAndSolvers.Helpers
using OperatorsAndSolvers.AbstractTypes
import OperatorsAndSolvers.AbstractTypes: dim_domain
import OperatorsAndSolvers.AbstractTypes: dim_range
import OperatorsAndSolvers.AbstractTypes: residual!
import OperatorsAndSolvers.AbstractTypes: allocate_jacobian
import OperatorsAndSolvers.AbstractTypes: jacobian!
import OperatorsAndSolvers.AbstractTypes: directional_jacobian!
import OperatorsAndSolvers.AbstractTypes: allocate_cache
import OperatorsAndSolvers.AbstractTypes: allocate_subcache
import OperatorsAndSolvers.AbstractTypes: solve!
using OperatorsAndSolvers.LineSearches

include("OperatorsAndSolvers.jl")
export AbstractSystemOperator
export AbstractLinearSystemOperator
export get_matrix
export get_vector
export LinearSystemOperator

export AbstractSystemSolver

include("DirectSolvers.jl")
export BackslashSolver
export LUSolver

include("AbstractIterativeSystemSolver.jl")
export IterativeSolverFlag
export isconverged

export AbstractIterativeSystemSolver
export get_config
export update_direction!

export IterativeSystemSolverCache
export reset_cache!

include("IterativeSystemSolverConfig.jl")
export IterativeSystemSolverConfig
export get_maxiter
export get_convergence_state
export fastnorm

include("IterativeSystemSolverLineSearch.jl")
export LineSearchOperatorForIterativeSystemSolver
export update_lsop!

include("IterativeSolvers/NewtonRaphson.jl")
export NewtonRaphsonSolver

include("IterativeSolvers/GradientDescent.jl")
export GradientDescentSolver

include("IterativeSolvers/ConjugateGradient.jl")
export ConjugateGradientSolver

end # module Systems
