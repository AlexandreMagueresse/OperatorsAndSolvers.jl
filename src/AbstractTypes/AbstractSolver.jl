##################
# AbstractSolver #
##################
"""
    AbstractSolver

Abstract type for all solvers, i.e. procedures that minimise the norm of the
residual of an operator.

# Mandatory methods
- `allocate_cache`
- `solve!`

# Optional methods
- `allocate_subcache`
- `allocate_initial_guess`

# Usage
    # This call will allocate the cache and an initial guess
    us, cache = solve(sv, op, Float32)

    # This call will allocate the cache and start from `us`
    us, cache = solve!(us, sv, op)

    # This call will reuse the cache and start from `us`
    us, cache = solve!(us, sv, op, cache)
"""
abstract type AbstractSolver end

"""
    allocate_cache(
      sv::AbstractSolver, op::AbstractOperator{N},
      us::NTuple{N,AbstractVector}
    ) -> AbstractSolverCache

Allocate the cache of the solver.
"""
function allocate_cache(
  sv::AbstractSolver, op::AbstractOperator{N},
  us::NTuple{N,AbstractVector}
) where {N}
  @abstractmethod
end

"""
    allocate_subcache

Allocate the subcache of the solver, if it belongs to a family of solvers that
share a common cache.
"""
function allocate_subcache end

"""
    allocate_initial_guess(
      sv::AbstractSolver, op::AbstractOperator,
      ::Type{T}
    ) -> AbstractVector

Allocate an initial guess with type `T`.

Default to a dense vector filled with zeros.
"""
function allocate_initial_guess(
  sv::AbstractSolver, op::AbstractOperator{N},
  ::Type{T}
) where {N,T}
  us = ()
  for k in 1:N
    n = dim_domain(op, Val(k))
    uk = zeros(T, n)
    us = (us..., uk)
  end
  us
end

"""
    solve(
      sv::AbstractSolver, op::AbstractOperator,
      ::Type{T}; kwargs...
    ) -> (AbstractVector, AbstractSolverCache)

Allocate an initial guess and solve the operator.
"""
function solve(
  sv::AbstractSolver, op::AbstractOperator,
  ::Type{T}; kwargs...
) where {T}
  us = allocate_initial_guess(sv, op, T)
  solve!(us, sv, op; kwargs...)
end


"""
    solve!(
      us::NTuple{N,AbstractVector}, sv::AbstractSolver,
      op::AbstractOperator{N}; kwargs...
    ) -> (AbstractVector, AbstractSolverCache)

Allocate the cache and solve the operator starting from `us`.

The arguments `us` will be updated in place. The cache is meant to be reused in
subsequent solves.
"""
function solve!(
  us::NTuple{N,AbstractVector}, sv::AbstractSolver,
  op::AbstractOperator{N}; kwargs...
) where {N}
  cache = allocate_cache(sv, op, us)
  solve!(us, sv, op, cache; kwargs...)
end

"""
    solve!(
      us::NTuple{N,AbstractVector}, sv::AbstractSolver,
      op::AbstractOperator{N}, cache::AbstractSolverCache; kwargs...
    ) -> (AbstractVector, AbstractSolverCache)

Solve the operator starting from `us` using the cache from a previous solve.
"""
function solve!(
  us::NTuple{N,AbstractVector}, sv::AbstractSolver,
  op::AbstractOperator{N}, cache::AbstractSolverCache; kwargs...
) where {N}
  @abstractmethod
end
