#################
# AbstractTypes #
#################
using OperatorsAndSolvers.AbstractTypes

export AbstractOperatorType
export NonlinearOperatorType
export AbstractQuasilinearOperatorType
export QuasilinearOperatorType
export AbstractSemilinearOperatorType
export SemilinearOperatorType
export AbstractLinearOperatorType
export LinearOperatorType

export AbstractOperator
export OperatorType
export AbstractNonlinearOperator
export AbstractQuasilinearOperator
export AbstractSemilinearOperator
export AbstractLinearOperator

export dim_domain
export dim_range
export allocate_zero
export allocate_residual
export residual
export residual!
export allocate_jacobian
export jacobian
export jacobian!
export allocate_directional_jacobian
export directional_jacobian
export directional_jacobian!

export AbstractSolverCache

export AbstractSolver

export allocate_cache
export allocate_subcache
export allocate_initial_guess
export solve
export solve!

################
# LineSearches #
################
using OperatorsAndSolvers.LineSearches

export AbstractLineSearchOperator
export get_ϕ
export get_dϕ
export get_ϕ0
export get_dϕ0
export LineSearchOperator

export AbstractLineSearchSolver

export ConstantStepper
export ExponentialStepper

###########
# Systems #
###########
using OperatorsAndSolvers.Systems

export AbstractSystemOperator
export AbstractLinearSystemOperator
export get_matrix
export get_vector
export LinearSystemOperator

export AbstractSystemSolver

export BackslashSolver
export LUSolver

export IterativeSolverFlag
export isconverged

export AbstractIterativeSystemSolver
export get_config
export update_direction!

export IterativeSystemSolverCache

export IterativeSystemSolverConfig
export get_maxiter
export get_convergence_state
export fastnorm

export LineSearchOperatorForIterativeSystemSolver
export update_lsop!

export NewtonRaphsonSolver
export GradientDescentSolver

########
# ODEs #
########
using OperatorsAndSolvers.ODEs

export AbstractODEOperator
export residual
export residual!
export allocate_jacobian_U
export jacobian_U
export jacobian_U!
export allocate_jacobian_U̇
export jacobian_U̇
export jacobian_U̇!
export allocate_directional_jacobian_U
export directional_jacobian_U
export directional_jacobian_U!
export allocate_directional_jacobian_U̇
export directional_jacobian_U̇
export directional_jacobian_U̇!
export AbstractQuasilinearODEOperator
export AbstractSemilinearODEOperator
export AbstractLinearODEOperator

export AbstractFormulation
export Formulation_U
export Formulation_U̇
export AbstractODESolver
export ODESolverCache

export ExplicitEulerSolver
