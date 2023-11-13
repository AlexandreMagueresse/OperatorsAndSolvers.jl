#######################
# NewtonRaphsonSolver #
#######################
"""
    NewtonRaphsonSolver

Newton-Raphson iterative solver for systems of equations.
"""
struct NewtonRaphsonSolver{SV,C} <:
       AbstractIterativeSystemSolver
  lsv::SV
  config::C
end

get_config(sv::NewtonRaphsonSolver) = sv.config

############################
# NewtonRaphsonSolverCache #
############################
"""
    NewtonRaphsonSolverCache

Cache corresponding to `NewtonRaphsonSolver`.
"""
struct NewtonRaphsonSolverCache{J,C} <:
       AbstractSolverCache
  J::J
  lcache::C
end

# IterativeSystemSolver interface
function allocate_subcache(
  sv::NewtonRaphsonSolver, op::AbstractSystemOperator,
  us::NTuple{1,AbstractVector}, r::AbstractVector, Δ::AbstractVector
)
  u, = us
  T = eltype(u)

  # J = allocate_jacobian(op, Val(1), T)
  J = jacobian(op, Val(1), us)

  lsv = sv.lsv
  lop = LinearSystemOperator(J, r)
  lus = (Δ,)
  lcache = allocate_cache(lsv, lop, lus)

  NewtonRaphsonSolverCache(J, lcache)
end

function update_direction!(
  Δ::AbstractVector, sv::NewtonRaphsonSolver,
  op::AbstractSystemOperator, cache::NewtonRaphsonSolverCache,
  us::NTuple{1,AbstractVector}, r::AbstractVector
)
  lsv = sv.lsv
  J = cache.J
  lcache = cache.lcache

  # Update jacobian
  jacobian!(J, op, Val(1), us)

  # Build linear system
  lop = LinearSystemOperator(J, r)
  lus = (Δ,)
  lus, lcache = solve!(lus, lsv, lop, lcache)
  Δ, = lus

  Δ
end
