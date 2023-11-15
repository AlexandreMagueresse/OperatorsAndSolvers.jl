module ODEs

using LinearAlgebra
using StaticArrays

using OperatorsAndSolvers.Helpers
using OperatorsAndSolvers.AbstractTypes
import OperatorsAndSolvers.AbstractTypes: dim_domain
import OperatorsAndSolvers.AbstractTypes: dim_range
import OperatorsAndSolvers.AbstractTypes: residual
import OperatorsAndSolvers.AbstractTypes: residual!
import OperatorsAndSolvers.AbstractTypes: jacobian
import OperatorsAndSolvers.AbstractTypes: jacobian!
import OperatorsAndSolvers.AbstractTypes: directional_jacobian
import OperatorsAndSolvers.AbstractTypes: directional_jacobian!
import OperatorsAndSolvers.AbstractTypes: allocate_cache
import OperatorsAndSolvers.AbstractTypes: allocate_subcache
import OperatorsAndSolvers.AbstractTypes: solve
import OperatorsAndSolvers.AbstractTypes: solve!
using OperatorsAndSolvers.Systems

include("Utils.jl")
export _make_us
export _make_u̇!
export _make_u!

include("OperatorsAndSolvers.jl")
export AbstractODEOperator
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

export AbstractFormulationType
export UFormulationType
export U̇FormulationType
export AbstractODESolverType
export ImplicitODESolverType
export ExplicitODESolverType
export AbstractODESolver
export FormulationType
export ODESolverType
export ODESolverCache

include("ExplicitEuler.jl")
export ExplicitEulerSolver

include("ImplicitEuler.jl")
export ImplicitEulerSolver

end # module ODEs
