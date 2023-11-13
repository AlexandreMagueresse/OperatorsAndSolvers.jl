module AbstractTypes

using OperatorsAndSolvers.Helpers

include("AbstractOperatorType.jl")
export AbstractOperatorType
export NonlinearOperatorType
export AbstractQuasilinearOperatorType
export QuasilinearOperatorType
export AbstractSemilinearOperatorType
export SemilinearOperatorType
export AbstractLinearOperatorType
export LinearOperatorType

include("AbstractOperator.jl")
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

include("AbstractSolverCache.jl")
export AbstractSolverCache

include("AbstractSolver.jl")
export AbstractSolver

export allocate_cache
export allocate_initial_guess
export solve
export solve!

end # module AbstractTypes
