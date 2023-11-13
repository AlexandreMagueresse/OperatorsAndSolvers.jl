@enum IterativeSolverFlag begin
  CONVERGED_ATOL
  CONVERGED_RTOL
  NOT_CONVERGED
  DIVERGED_ITER
end

"""
    isconverged(flag::IterativeSolverFlag)

Tell whether an `IterativeSolverFlag` corresponds to a converged state.
"""
function isconverged(flag::IterativeSolverFlag)
  (flag == CONVERGED_ATOL) || (flag == CONVERGED_RTOL)
end

#################################
# AbstractIterativeSystemSolver #
#################################
"""
    AbstractIterativeSystemSolver

Abstract type for iterative solvers of systems of equations.

# Mandatory methods
- `get_config`
- `allocate_subcache`
- `update_direction!`
"""
abstract type AbstractIterativeSystemSolver <:
              AbstractSolver end

"""
    get_config(sv::AbstractIterativeSystemSolver) -> IterativeSystemSolverConfig

Return the configuration of the iterative solver.
"""
function get_config(sv::AbstractIterativeSystemSolver)
  @abstractmethod
end

"""
  allocate_subcache(
    sv::AbstractIterativeSystemSolver, op::AbstractSystemOperator,
    us::NTuple{1,AbstractVector}, r::AbstractVector, Δ::AbstractVector
  ) -> AbstractSolverCache

Allocate the subcache of the iterative solver.
"""
function allocate_subcache(
  sv::AbstractIterativeSystemSolver, op::AbstractSystemOperator,
  us::NTuple{1,AbstractVector}, r::AbstractVector, Δ::AbstractVector
)
  @abstractmethod
end

"""
    update_direction!(
      Δ::AbstractVector, sv::AbstractIterativeSystemSolver,
      op::AbstractSystemOperator, cache::AbstractSolverCache,
      us::NTuple{1,AbstractVector}, r::AbstractVector
    ) -> AbstractVector

Update the direction according to update rule of the iterative solver.
"""
function update_direction!(
  Δ::AbstractVector, sv::AbstractIterativeSystemSolver,
  op::AbstractSystemOperator, cache::AbstractSolverCache,
  us::NTuple{1,AbstractVector}, r::AbstractVector
)
  @abstractmethod
end

##############################
# IterativeSystemSolverCache #
##############################
"""
    IterativeSystemSolverCache

Cache corresponding to an `AbstractIterativeSystemSolver`.
"""
struct IterativeSystemSolverCache{R,Δ,F,C} <:
       AbstractSolverCache
  r::R
  Δ::Δ
  flag::F
  subcache::C
end

"""
    isconverged(cache::IterativeSystemSolverCache)

Tell whether the flag stored in an `IterativeSystemSolverCache` corresponds to
a converged state.
"""
function isconverged(cache::IterativeSystemSolverCache)
  isconverged(IterativeSolverFlag(cache.flag[]))
end

# AbstractSolver interface
function allocate_cache(
  sv::AbstractIterativeSystemSolver, op::AbstractSystemOperator,
  us::NTuple{1,AbstractVector}
)
  u, = us
  T = eltype(u)

  r = allocate_residual(op, T)
  Δ, = allocate_zero(op, T)
  flag = Ref(Int(NOT_CONVERGED))

  subcache = allocate_subcache(sv, op, us, r, Δ)

  IterativeSystemSolverCache(r, Δ, flag, subcache)
end

function solve!(
  us::NTuple{1,AbstractVector}, sv::AbstractIterativeSystemSolver,
  op::AbstractSystemOperator, cache::IterativeSystemSolverCache
)
  u, = us

  r, Δ = cache.r, cache.Δ
  subcache = cache.subcache

  config = get_config(sv)
  maxiter = get_maxiter(config)

  # Compute residual and check convergence
  us = (u,)
  residual!(r, op, us)
  flag, res = get_convergence_state(config, r)

  # Iterate until converged or maximum number of iterations
  initial_res = res
  iter = 0
  while !isconverged(flag) && (iter <= maxiter)
    iter += 1

    # Update direction
    update_direction!(Δ, sv, op, subcache, us, r)

    # Update u
    @inbounds @simd for i in eachindex(u, Δ)
      u[i] -= Δ[i]
    end

    # Compute residual and check convergence
    us = (u,)
    residual!(r, op, us)
    flag, res = get_convergence_state(config, r, initial_res)
  end

  if iter > maxiter
    cache.flag[] = Int(DIVERGED_ITER)
  else
    cache.flag[] = Int(flag)
  end

  us, cache
end
