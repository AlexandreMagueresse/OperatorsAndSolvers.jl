#######################
# ExplicitEulerSolver #
#######################
"""
    ExplicitEulerSolver

Euler's explicit scheme.
Type          explicit
Order         1
Description   res(t₋, u₋, k) = 0
"""
struct ExplicitEulerSolver{F,S} <:
       AbstractODESolver{F}
  subsv::S

  function ExplicitEulerSolver{F}(subsv) where {F}
    S = typeof(subsv)
    new{F,S}(subsv)
  end
end

ExplicitEulerSolver(subsv) = ExplicitEulerSolver{Formulation_U}(subsv)

"""
    ExplicitEulerSolverCache

Cache corresponding to an `ExplicitEulerSolver`.
"""
struct ExplicitEulerSolverCache{C} <:
       AbstractSolverCache
  subcache::C
end

# AbstractODESolver interface
function allocate_subcache(
  sv::ExplicitEulerSolver, op::AbstractODEOperator{N},
  t₋::Real, dt::Real, u₋::AbstractVector,
  u̇_temp::AbstractVector, r_temp::AbstractVector, j_temp::AbstractVector
) where {N}
  dt⁻¹ = inv(dt)
  subop = ForwardEulerODEOperator(t₋, u₋, dt⁻¹, u̇_temp, op)
  subsv = sv.subsv
  subus = (r_temp,)
  subcache = allocate_cache(subsv, subop, subus)
  ExplicitEulerSolverCache(subcache)
end

function solve!(
  u₊::AbstractVector, sv::ExplicitEulerSolver{Formulation_U},
  op::AbstractODEOperator, t₋::Real, dt::Real, u₋::AbstractVector,
  cache::ODESolverCache
)
  t₊ = t₋ + dt
  dt⁻¹ = inv(dt)
  u̇_temp = cache.u̇_temp

  subop = ForwardEulerODEOperator(t₋, u₋, dt⁻¹, u̇_temp, op)
  subsv = sv.subsv
  subcache = cache.subcache.subcache

  us = (u₊,)
  us, subcache = solve!(us, subsv, subop, subcache)
  if !isconverged(subcache)
    msg = "$(string(typeof(subsv).name.name)) did not converge, aborting."
    throw(msg)
  end
  u₊, = us

  (t₊, dt, u₊), cache
end

struct ForwardEulerODEOperator{N,T,U,D,U̇,O} <:
       AbstractSystemOperator{N,N,NonlinearOperatorType}
  t₋::T
  u₋::U
  dt⁻¹::D
  u̇_temp::U̇
  odeop::O

  function ForwardEulerODEOperator(t₋, u₋, dt⁻¹, u̇_temp, op)
    N = length(u₋)
    T = typeof(t₋)
    U = typeof(u₋)
    D = typeof(dt⁻¹)
    U̇ = typeof(u̇_temp)
    O = typeof(op)
    new{N,T,U,D,U̇,O}(t₋, u₋, dt⁻¹, u̇_temp, op)
  end
end

function residual!(
  r::AbstractVector, op::ForwardEulerODEOperator,
  us::NTuple{1,AbstractVector}
)
  t₋, u₋, dt⁻¹, u̇_temp = op.t₋, op.u₋, op.dt⁻¹, op.u̇_temp
  odeop = op.odeop

  u, = us
  _make_u̇!(u̇_temp, u, u₋, dt⁻¹)

  residual!(r, odeop, t₋, u₋, u̇_temp)
  r
end

function jacobian!(
  J::AbstractMatrix, op::ForwardEulerODEOperator, k::Val{1},
  us::NTuple{1,AbstractVector}
)
  t₋, u₋, dt⁻¹, u̇_temp = op.t₋, op.u₋, op.dt⁻¹, op.u̇_temp
  odeop = op.odeop

  u, = us
  _make_u̇!(u̇_temp, u, u₋, dt⁻¹)

  jacobian_U̇!(J, odeop, t₋, u₋, u̇_temp)
  rmul!(J, dt⁻¹)
  J
end

function directional_jacobian!(
  j::AbstractVector, J, op::ForwardEulerODEOperator, k::Val{1},
  us::NTuple{1,AbstractVector}, v::AbstractVector
)
  t₋, u₋, dt⁻¹, u̇_temp = op.t₋, op.u₋, op.dt⁻¹, op.u̇_temp
  odeop = op.odeop

  u, = us
  _make_u̇!(u̇_temp, u, u₋, dt⁻¹)

  directional_jacobian_U̇!(j, J, odeop, t₋, u₋, u̇_temp, v)
  rmul!(j, dt⁻¹)
  J
end

# function odesolve!(
#   ::Type{<:AbstractODEProblem}, ::Type{<:Formulation_U̇},
#   odesolver::ExplicitEuler, t₋, u₋, dt, u₊
# )
#   problem = odesolver.problem
#   nlsolver = odesolver.nlsolver

#   t₊ = t₋ + dt

#   function nl_res!(r, k)
#     res!(problem, r, t₋, u₋, k)
#     r
#   end

#   function nl_jac_vec!(j, k, vec)
#     jac_u̇_vec!(problem, j, t₋, u₋, k, vec)
#     j
#   end

#   k = u₊
#   fill!(k, 0)
#   nlsolve!(nlsolver, k, nl_res!, nl_jac_vec!)
#   axpby!(1, u₋, dt, k)
#   u₊ = k

#   t₊, u₊
# end

# function odesolve!(
#   ::Type{<:AbstractIsolatedProblem},
#   odesolver::ExplicitEuler, t₋, u₋, dt, u₊
# )
#   problem = odesolver.problem
#   lsolver = get_lsolver(problem)
#   mat = get_mat(lsolver)
#   vec = get_vec(lsolver)
#   t₊ = t₋ + dt

#   lhs!(problem, mat, t₋, u₋)
#   rhs!(problem, vec, t₋, u₋)

#   k = u₊
#   lsolve!(lsolver, u₊)
#   axpby!(1, u₋, dt, k)
#   u₊ = k

#   t₊, u₊
# end
