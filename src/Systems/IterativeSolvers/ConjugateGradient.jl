###########################
# ConjugateGradientSolver #
###########################
"""
    ConjugateGradientSolver

Conjugate gradient iterative solver for systems of equations.
"""
struct ConjugateGradientSolver{C} <:
       AbstractIterativeSystemSolver
  config::C
end

get_config(sv::ConjugateGradientSolver) = sv.config

################################
# ConjugateGradientSolverCache #
################################
"""
    ConjugateGradientSolverCache

Cache corresponding to `ConjugateGradientSolver`.
"""
struct ConjugateGradientSolverCache{R,RR,D,Ad} <:
       AbstractSolverCache
  r::R
  rr::RR
  d::D
  Ad::Ad
end

# IterativeSystemSolver interface
function allocate_subcache(
  sv::ConjugateGradientSolver, op::AbstractSystemOperator,
  us::NTuple{1,AbstractVector}, r::AbstractVector, Δ::AbstractVector
)
  u, = us
  T = eltype(u)

  r = allocate_residual(op, T)
  residual!(r, op, us)
  rmul!(r, -1)
  rr = Ref(dot(r, r))
  d = copy(r)
  Ad = allocate_residual(op, T)

  ConjugateGradientSolverCache(r, rr, d, Ad)
end

function reset_subcache!(
  cache::ConjugateGradientSolverCache, sv::ConjugateGradientSolver,
  op::AbstractLinearSystemOperator, us::NTuple{1,AbstractVector}
)
  r, d = cache.r, cache.d

  residual!(r, op, us)
  rmul!(r, -1)
  cache.rr[] = dot(r, r)
  copy!(d, r)

  cache
end

function update_direction!(
  Δ::AbstractVector, sv::ConjugateGradientSolver,
  op::AbstractSystemOperator, cache::ConjugateGradientSolverCache,
  us::NTuple{1,AbstractVector}, r::AbstractVector
)
  # TODO
  # This involves a line search and the choice of β, including
  # - Fletcher-Reeves
  # - Polak-Ribiere
  # - Hestenes-Stiefel
  # - Dai-Yuan
end

function update_direction!(
  Δ::AbstractVector, sv::ConjugateGradientSolver,
  op::AbstractLinearSystemOperator, cache::ConjugateGradientSolverCache,
  us::NTuple{1,AbstractVector}, r::AbstractVector
)
  r, rr, d, Ad = cache.r, cache.rr[], cache.d, cache.Ad
  A = get_matrix(op)
  b = get_vector(op)

  # Compute α
  mul!(Ad, A, d)
  α = rr / dot(d, Ad)

  # Copy into Δ
  copy!(Δ, -d)
  rmul!(Δ, α)

  # Update residual
  r .-= α .* Ad

  # Compute β
  γ = dot(r, r)
  β = γ / rr
  cache.rr[] = γ

  # Update direction
  d .= r .+ β .* d

  Δ
end
