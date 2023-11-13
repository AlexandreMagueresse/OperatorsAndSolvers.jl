###################
# BackslashSolver #
###################
"""
    BackslashSolver

Linear system solver that calls the backslash operator `\\` native to Julia.
"""
struct BackslashSolver <:
       AbstractSystemSolver end

"""
    BackslashSolverCache

Cache corresponding to `BackslashSolver`.
"""
mutable struct BackslashSolverCache{R,J} <:
               AbstractSolverCache
  r::R
  J::J
end

# AbstractSolver interface
function allocate_cache(
  sv::BackslashSolver, op::AbstractLinearSystemOperator,
  us::NTuple{1,AbstractVector}
)
  u, = us
  T = eltype(u)

  r = allocate_residual(op, T)
  # J = allocate_jacobian(op, Val(1), T)
  J = jacobian(op, Val(1), us)

  BackslashSolverCache(r, J)
end

function solve!(
  us::NTuple{1,AbstractVector}, sv::BackslashSolver,
  op::AbstractLinearSystemOperator, cache::BackslashSolverCache;
  update_residual::Bool=true, update_jacobian::Bool=false
)
  u, = us
  if update_residual
    copy!(cache.r, get_vector(op))
  end
  if update_jacobian
    copy!(cache.J, get_matrix(op))
  end

  copy!(u, cache.J \ cache.r)

  us = (u,)
  (us, cache)
end

############
# LUSolver #
############
"""
    LUSolver

Linear system solver that performs an LU decomposition of the matrix and reuses
it until it is updated.
"""
struct LUSolver <:
  AbstractSystemSolver end

"""
    LUSolverCache

Cache corresponding to `LUSolver`.
"""
mutable struct LUSolverCache{R,J,F} <:
   AbstractSolverCache
  r::R
  J::J
  factors::F
end

# AbstractSolver interface
function allocate_cache(
  sv::LUSolver, op::AbstractLinearSystemOperator,
  us::NTuple{1,AbstractVector}
)
  u, = us
  T = eltype(u)

  r = allocate_residual(op, T)
  # J = allocate_jacobian(op, Val(1), T)
  J = jacobian(op, Val(1), us)
  factors = lu(J)

  LUSolverCache(r, J, factors)
end

function solve!(
  us::NTuple{1,AbstractVector}, sv::LUSolver,
  op::AbstractLinearSystemOperator, cache::LUSolverCache;
  update_residual::Bool=true, update_jacobian::Bool=false
)
  u, = us
  if update_residual
    copy!(cache.r, get_vector(op))
  end
  if update_jacobian
    copy!(cache.J, get_matrix(op))
    cache.factors = lu(cache.J)
  end

  ldiv!(u, cache.factors, cache.r)

  us = (u,)
  (us, cache)
end
