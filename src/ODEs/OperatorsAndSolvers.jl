#######################
# AbstractODEOperator #
#######################
"""
    AbstractODEOperator{N,T}

Abstract type for ODE operators of the form ``f(t, u, u̇) = 0`` involving `N`
unknowns, of type `T`.
"""
abstract type AbstractODEOperator{N,T} <:
              AbstractOperator{3,T} end

# AbstractOperator interface
dim_domain(op::AbstractODEOperator, k::Val{1}) = 1
dim_domain(op::AbstractODEOperator{N}, k::Val{2}) where {N} = N
dim_domain(op::AbstractODEOperator{N}, k::Val{3}) where {N} = N

dim_range(op::AbstractODEOperator{N}) where {N} = N

"""
    residual(
      op::AbstractODEOperator,
      t::Real, u::AbstractVector, u̇::AbstractVector
    ) -> AbstractVector

Allocate and evaluate the residual vector of the ODE operator at `(t, u, u̇)`.
"""
function residual(
  op::AbstractODEOperator,
  t::Real, u::AbstractVector, u̇::AbstractVector
)
  T = eltype(t)
  r = allocate_residual(op, T)
  us = _make_us(t, u, u̇)
  residual!(r, op, us)
  r
end

"""
    residual!(
      r::AbstractVector, op::AbstractODEOperator,
      t::Real, u::AbstractVector, u̇::AbstractVector
    ) -> AbstractVector

Evaluate the residual vector of the ODE operator at `(t, u, u̇)`.
"""
function residual!(
  r::AbstractVector, op::AbstractODEOperator,
  t::Real, u::AbstractVector, u̇::AbstractVector
)
  us = _make_us(t, u, u̇)
  residual!(r, op, us)
end

"""
    allocate_jacobian_U(op::AbstractODEOperator, ::Type{T}) -> AbstractMatrix

Allocate the jacobian matrix of the ODE operator with respect to `U`.

Default to a dense matrix filled with zeros.
"""
function allocate_jacobian_U(op::AbstractODEOperator, ::Type{T}) where {T}
  allocate_jacobian(op, Val(2), T)
end

"""
    jacobian_U(
      op::AbstractODEOperator,
      t::Real, u::AbstractVector, u̇::AbstractVector
    ) -> AbstractMatrix

Allocate and evaluate the jacobian matrix of the ODE operator with respect to
`U` at `(t, u, u̇)`.
"""
function jacobian_U(
  op::AbstractODEOperator,
  t::Real, u::AbstractVector, u̇::AbstractVector
)
  us = _make_us(t, u, u̇)
  jacobian(op, Val(2), us)
end

"""
    jacobian_U!(
      J::AbstractMatrix, op::AbstractODEOperator,
      t::Real, u::AbstractVector, u̇::AbstractVector
    ) -> AbstractMatrix

Evaluate the jacobian matrix of the ODE operator with respect to `U` at
`(t, u, u̇)`.
"""
function jacobian_U!(
  J::AbstractMatrix, op::AbstractODEOperator,
  t::Real, u::AbstractVector, u̇::AbstractVector
)
  us = _make_us(t, u, u̇)
  jacobian!(J, op, Val(2), us)
end

"""
    allocate_jacobian_U̇(op::AbstractODEOperator, ::Type{T}) -> AbstractMatrix

Allocate the jacobian matrix of the ODE operator with respect to `U̇`.

Default to a dense matrix filled with zeros.
"""
function allocate_jacobian_U̇(op::AbstractODEOperator, ::Type{T}) where {T}
  allocate_jacobian(op, Val(3), T)
end

"""
    jacobian_U̇(
      op::AbstractODEOperator,
      t::Real, u::AbstractVector, u̇::AbstractVector
    ) -> AbstractMatrix

Allocate and evaluate the jacobian matrix of the ODE operator with respect to
`U̇` at `(t, u, u̇)`.
"""
function jacobian_U̇(
  op::AbstractODEOperator,
  t::Real, u::AbstractVector, u̇::AbstractVector
)
  us = _make_us(t, u, u̇)
  jacobian(op, Val(3), us)
end

"""
    jacobian_U̇!(
      J::AbstractMatrix, op::AbstractODEOperator,
      t::Real, u::AbstractVector, u̇::AbstractVector
    ) -> AbstractMatrix

Evaluate the jacobian matrix of the ODE operator with respect to `U̇` at
`(t, u, u̇)`.
"""
function jacobian_U̇!(
  J::AbstractMatrix, op::AbstractODEOperator,
  t::Real, u::AbstractVector, u̇::AbstractVector
)
  us = _make_us(t, u, u̇)
  jacobian!(J, op, Val(3), us)
end

"""
    allocate_directional_jacobian_U(
      op::AbstractOperator, ::Type{T}
    ) -> AbstractVector

Allocate the jacobian vector product of the operator with respect to `U`.

Default to a dense vector filled with zeros.
"""
function allocate_directional_jacobian_U(
  op::AbstractOperator, ::Type{T}
) where {T}
  allocate_directional_jacobian(op, Val(2), T)
end

"""
    directional_jacobian_U(
      op::AbstractODEOperator,
      t::Real, u::AbstractVector, u̇::AbstractVector, v::AbstractVector
    ) -> AbstractVector

Allocate and evaluate the jacobian of the ODE operator with respect to `U` at
`(t, u, u̇)` in the direction of `v`.
"""
function directional_jacobian_U(
  op::AbstractODEOperator,
  t::Real, u::AbstractVector, u̇::AbstractVector, v::AbstractVector
)
  us = _make_us(t, u, u̇)
  directional_jacobian(op, Val(2), us, v)
end

"""
    directional_jacobian_U!(
      j::AbstractVector, J, op::AbstractODEOperator,
      t::Real, u::AbstractVector, u̇::AbstractVector, v::AbstractVector
    ) -> AbstractVector

Evaluate the jacobian of the ODE operator with respect to `U` at `(t, u, u̇)` in
the direction of `v`.

Default to the standard matrix product `j = jacobian_U(op, t, u, u̇) ⋅ v`. Here
`J` is provided as a cache to compute the full jacobian matrix if needed, but
efficient implementations should not make use of it.
"""
function directional_jacobian_U!(
  j::AbstractVector, J, op::AbstractODEOperator,
  t::Real, u::AbstractVector, u̇::AbstractVector, v::AbstractVector
)
  us = _make_us(t, u, u̇)
  directional_jacobian!(j, J, op, Val(2), us, v)
end

"""
    allocate_directional_jacobian_U̇(
      op::AbstractOperator, ::Type{T}
    ) -> AbstractVector

Allocate the jacobian vector product of the operator with respect to `U̇`.

Default to a dense vector filled with zeros.
"""
function allocate_directional_jacobian_U̇(
  op::AbstractOperator, ::Type{T}
) where {T}
  allocate_directional_jacobian(op, Val(3), T)
end

"""
    directional_jacobian_U̇(
      op::AbstractODEOperator,
      t::Real, u::AbstractVector, u̇::AbstractVector, v::AbstractVector
    ) -> AbstractVector

Allocate and evaluate the jacobian of the ODE operator with respect to `U̇` at
`(t, u, u̇)` in the direction of `v`.
"""
function directional_jacobian_U̇(
  op::AbstractODEOperator,
  t::Real, u::AbstractVector, u̇::AbstractVector, v::AbstractVector
)
  us = _make_us(t, u, u̇)
  directional_jacobian(op, Val(3), us, v)
end

"""
    directional_jacobian_U̇!(
      j::AbstractVector, J, op::AbstractODEOperator,
      t::Real, u::AbstractVector, u̇::AbstractVector, v::AbstractVector
    ) -> AbstractVector

Evaluate the jacobian of the ODE operator with respect to `U̇` at `(t, u, u̇)` in
the direction of `v`.

Default to the standard matrix product `j = jacobian_U(op, t, u, u̇) ⋅ v`. Here
`J` is provided as a cache to compute the full jacobian matrix if needed, but
efficient implementations should not make use of it.
"""
function directional_jacobian_U̇!(
  j::AbstractVector, J, op::AbstractODEOperator,
  t::Real, u::AbstractVector, u̇::AbstractVector, v::AbstractVector
)
  us = _make_us(t, u, u̇)
  directional_jacobian!(j, J, op, Val(3), us, v)
end

##################################
# AbstractQuasilinearODEOperator #
##################################
"""
    AbstractQuasilinearODEOperator{N}

Abstract type for quasilinear ODE operators, i.e. ``M(t, u) u̇ = f(t, u)``. Alias
for `AbstractODEOperator{N,QuasilinearOperatorType}`.
"""
const AbstractQuasilinearODEOperator{N} =
  AbstractODEOperator{N,QuasilinearOperatorType}

#################################
# AbstractSemilinearODEOperator #
#################################
"""
    AbstractSemilinearODEOperator{N}

Abstract type for semilinear ODE operators, i.e. ``M(t) u̇ = f(t, u)``. Alias
for `AbstractODEOperator{N,SemilinearOperatorType}`.
"""
const AbstractSemilinearODEOperator{N} =
  AbstractODEOperator{N,SemilinearOperatorType}

#############################
# AbstractLinearODEOperator #
#############################
"""
    AbstractLinearODEOperator{N}

Abstract type for linear ODE operators, i.e. ``M(t) u̇ = K(t) u + f(t)``. Alias
for `AbstractODEOperator{N,LinearOperatorType}`.
"""
const AbstractLinearODEOperator{N} =
  AbstractODEOperator{N,LinearOperatorType}

#######################
# AbstractFormulation #
#######################
"""
    AbstractFormulation

Abstract trait that encodes the formulation type of an ODE solver.
"""
abstract type AbstractFormulation end

"""
    Formulation_U

Trait of an ODE solver solving for U.
"""
struct Formulation_U <: AbstractFormulation end

"""
    Formulation_U̇

Trait of an ODE solver solving for U̇.
"""
struct Formulation_U̇ <: AbstractFormulation end

#####################
# AbstractODESolver #
#####################
"""
AbstractODESolver

Abstract type for ODE solvers.

# Mandatory methods
- `allocate_subcache`
"""
abstract type AbstractODESolver{F<:AbstractFormulation} <:
              AbstractSolver end

##################
# ODESolverCache #
##################
"""
    ODESolverCache

Cache corresponding to an `AbstractODESolver`.
"""
struct ODESolverCache{U̇,R,J,C} <:
       AbstractSolverCache
  u̇_temp::U̇
  r_temp::R
  j_temp::J
  subcache::C
end

"""
    allocate_subcache(
      sv::AbstractODESolver, op::AbstractODEOperator,
      t₋::Real, dt::Real, u₋::AbstractVector,
      u̇_temp::AbstractVector, r_temp::AbstractVector, j_temp::AbstractVector
    ) -> AbstractSolverCache

Allocate the subcache of the ODE solver.
"""
function allocate_subcache(
  sv::AbstractODESolver, op::AbstractODEOperator,
  t₋::Real, dt::Real, u₋::AbstractVector,
  u̇_temp::AbstractVector, r_temp::AbstractVector, j_temp::AbstractVector
)
  @abstractmethod
end

# AbstractSolver interface
"""
    allocate_cache(
      sv::AbstractODESolver, op::AbstractODEOperator,
      t₋::Real, dt::Real, u₋::AbstractVector,
    ) -> ODESolverCache

Allocate the cache of the ODE solver.
"""
function allocate_cache(
  sv::AbstractODESolver, op::AbstractODEOperator,
  t₋::Real, dt::Real, u₋::AbstractVector,
)
  T = eltype(t₋)

  u̇_temp = similar(u₋)
  r_temp = allocate_residual(op, T)
  j_temp = allocate_directional_jacobian_U(op, T)

  subcache = allocate_subcache(sv, op, t₋, dt, u₋, u̇_temp, r_temp, j_temp)

  ODESolverCache(u̇_temp, r_temp, j_temp, subcache)
end

"""
    solve(
      sv::AbstractODESolver, op::AbstractODEOperator,
      t₋::Real, dt::Real, u₋::AbstractVector; kwargs...
    ) -> ((Real, Real, AbstractVector), ODESolverCache)

Allocate a cache for `u₊` and solve the ODE operator between `t₋` and
`t₊ = t₋ + dt`, starting from `u₋`.
"""
function solve(
  sv::AbstractODESolver, op::AbstractODEOperator,
  t₋::Real, dt::Real, u₋::AbstractVector; kwargs...
)
  u₊ = copy(u₋)
  solve!(u₊, sv, op, t₋, dt, u₋; kwargs...)
end

"""
    solve!(
      u₊::AbstractVector, sv::AbstractODESolver,
      op::AbstractODEOperator, t₋::Real, dt::Real, u₋::AbstractVector; kwargs...
    ) -> ((Real, Real, AbstractVector), ODESolverCache)

Allocate the cache and solve the ODE operator between `t₋` and `t₊ = t₋ + dt`,
starting from `u₋`.

The argument `u₊` will be updated in place. The cache is meant to be reused in
subsequent solves.
"""
function solve!(
  u₊::AbstractVector, sv::AbstractODESolver,
  op::AbstractODEOperator, t₋::Real, dt::Real, u₋::AbstractVector; kwargs...
)
  cache = allocate_cache(sv, op, t₋, dt, u₋)
  solve!(u₊, sv, op, t₋, dt, u₋, cache; kwargs...)
end

"""
    function solve!(
      u₊::AbstractVector, sv::AbstractODESolver,
      op::AbstractODEOperator, t₋::Real, dt::Real, u₋::AbstractVector,
      cache::ODESolverCache; kwargs...
    ) -> ((Real, Real, AbstractVector), ODESolverCache)

Solve the ODE operator between `t₋` and `t₊ = t₋ + dt`, starting from `u₋`,
using the cache from a previous solve.
"""
function solve!(
  u₊::AbstractVector, sv::AbstractODESolver,
  op::AbstractODEOperator, t₋::Real, dt::Real, u₋::AbstractVector,
  cache::ODESolverCache; kwargs...
)
  @abstractmethod
end
