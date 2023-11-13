module LineSearches

using OperatorsAndSolvers.Helpers
using OperatorsAndSolvers.AbstractTypes
import OperatorsAndSolvers.AbstractTypes: dim_domain
import OperatorsAndSolvers.AbstractTypes: dim_range
import OperatorsAndSolvers.AbstractTypes: residual!
import OperatorsAndSolvers.AbstractTypes: jacobian!
import OperatorsAndSolvers.AbstractTypes: directional_jacobian!
import OperatorsAndSolvers.AbstractTypes: allocate_cache
import OperatorsAndSolvers.AbstractTypes: solve!

include("OperatorsAndSolvers.jl")
export AbstractLineSearchOperator
export get_ϕ
export get_dϕ
export get_ϕ0
export get_dϕ0
export LineSearchOperator

export AbstractLineSearchSolver

include("Steppers.jl")
export ConstantStepper
export ExponentialStepper

end # module LineSearches
