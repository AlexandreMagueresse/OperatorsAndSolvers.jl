#########################
# GradientDescentSolver #
#########################
"""
    GradientDescentSolver

Gradient descent iterative solver for systems of equations.
"""
struct GradientDescentSolver{LSSV,C} <:
       AbstractIterativeSystemSolver
  lssv::LSSV
  config::C
end

get_config(sv::GradientDescentSolver) = sv.config

##############################
# GradientDescentSolverCache #
##############################
"""
    GradientDescentSolverCache

Cache corresponding to `GradientDescentSolver`.
"""
struct GradientDescentSolverCache{JM,LSO,LSU,LSC,AΔ} <:
       AbstractSolverCache
  J::JM
  lsop::LSO
  lsus::LSU
  lscache::LSC
  AΔ::AΔ
end

# IterativeSystemSolver interface
function allocate_subcache(
  sv::GradientDescentSolver, op::AbstractSystemOperator,
  us::NTuple{1,AbstractVector}, r::AbstractVector, Δ::AbstractVector
)
  u, = us
  T = eltype(u)

  # J = allocate_jacobian(op, Val(1), T)
  J = jacobian(op, Val(1), us)
  lsop = LineSearchOperatorForIterativeSystemSolver(op, T)
  lssv = sv.lssv
  lsus = allocate_initial_guess(lssv, lsop, T)
  lscache = allocate_cache(lssv, lsop, lsus)

  if op isa LinearSystemOperator
    AΔ = allocate_residual(op, T)
  else
    AΔ = nothing
  end

  GradientDescentSolverCache(J, lsop, lsus, lscache, AΔ)
end

function update_direction!(
  Δ::AbstractVector, sv::GradientDescentSolver,
  op::AbstractSystemOperator, cache::GradientDescentSolverCache,
  us::NTuple{1,AbstractVector}, r::AbstractVector
)
  u, = us

  # Update directional jacobian
  J = cache.J
  directional_jacobian!(Δ, J, op, Val(1), us, r)

  # Line search
  lsop, lssv, lsus, lscache = cache.lsop, sv.lssv, cache.lsus, cache.lscache
  update_lsop!(lsop, u, Δ, r)
  lsus, lscache = solve!(lsus, lssv, lsop, lscache)
  lsu, = lsus
  α = lsu[1]
  rmul!(Δ, α)

  Δ
end

function update_direction!(
  Δ::AbstractVector, sv::GradientDescentSolver,
  op::AbstractLinearSystemOperator, cache::GradientDescentSolverCache,
  us::NTuple{1,AbstractVector}, r::AbstractVector
)
  # Update directional jacobian
  J = cache.J
  directional_jacobian!(Δ, J, op, Val(1), us, r)

  # Line search
  A = get_matrix(op)
  AΔ = cache.AΔ
  mul!(AΔ, A, Δ)
  α = dot(r, AΔ) / dot(AΔ, AΔ)
  rmul!(Δ, α)

  Δ
end
