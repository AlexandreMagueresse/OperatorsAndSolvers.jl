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
struct ExplicitEulerSolverCache{J,C} <:
       AbstractSolverCache
  J_temp::J
  subcache::C
end

#################
# Formulation_U #
#################
function allocate_subcache(
  F::Formulation_U, sv::ExplicitEulerSolver, op::AbstractODEOperator,
  t₋::Real, dt::Real, u₋::AbstractVector,
  u̇_temp::AbstractVector, r_temp::AbstractVector, j_temp::AbstractVector
)
  dt⁻¹ = inv(dt)

  subop = ExplicitEulerUOperator(t₋, u₋, dt⁻¹, u̇_temp, op)
  subsv = sv.subsv
  subus = (r_temp,)
  subcache = allocate_cache(subsv, subop, subus)

  ExplicitEulerSolverCache(nothing, subcache)
end

function solve!(
  u₊::AbstractVector, F::Formulation_U, sv::ExplicitEulerSolver,
  op::AbstractODEOperator, t₋::Real, dt::Real, u₋::AbstractVector,
  cache::ODESolverCache
)
  t₊ = t₋ + dt
  dt⁻¹ = inv(dt)
  u̇_temp = cache.u̇_temp

  subop = ExplicitEulerUOperator(t₋, u₋, dt⁻¹, u̇_temp, op)
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

struct ExplicitEulerUOperator{N,T,U,D,U̇,O} <:
       AbstractSystemOperator{N,N,NonlinearOperatorType}
  t₋::T
  u₋::U
  dt⁻¹::D
  u̇_temp::U̇
  odeop::O

  function ExplicitEulerUOperator(t₋, u₋, dt⁻¹, u̇_temp, op)
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
  r::AbstractVector, op::ExplicitEulerUOperator,
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
  J::AbstractMatrix, op::ExplicitEulerUOperator, k::Val{1},
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
  j::AbstractVector, J, op::ExplicitEulerUOperator, k::Val{1},
  us::NTuple{1,AbstractVector}, v::AbstractVector
)
  t₋, u₋, dt⁻¹, u̇_temp = op.t₋, op.u₋, op.dt⁻¹, op.u̇_temp
  odeop = op.odeop

  u, = us
  _make_u̇!(u̇_temp, u, u₋, dt⁻¹)

  directional_jacobian_U̇!(j, J, odeop, t₋, u₋, u̇_temp, v)
  rmul!(j, dt⁻¹)
  j
end

#################
# Formulation_U̇ #
#################
function allocate_subcache(
  F::Formulation_U̇, sv::ExplicitEulerSolver, op::AbstractODEOperator,
  t₋::Real, dt::Real, u₋::AbstractVector,
  u̇_temp::AbstractVector, r_temp::AbstractVector, j_temp::AbstractVector
)
  subop = ExplicitEulerU̇Operator(t₋, u₋, op)
  subsv = sv.subsv
  subus = (u̇_temp,)
  subcache = allocate_cache(subsv, subop, subus)

  ExplicitEulerSolverCache(nothing, subcache)
end

function solve!(
  u₊::AbstractVector, F::Formulation_U̇, sv::ExplicitEulerSolver,
  op::AbstractODEOperator, t₋::Real, dt::Real, u₋::AbstractVector,
  cache::ODESolverCache
)
  t₊ = t₋ + dt
  u̇_temp = cache.u̇_temp

  subop = ExplicitEulerU̇Operator(t₋, u₋, op)
  subsv = sv.subsv
  subcache = cache.subcache.subcache

  fill!(u̇_temp, 0)
  us = (u̇_temp,)
  us, subcache = solve!(us, subsv, subop, subcache)
  if !isconverged(subcache)
    msg = "$(string(typeof(subsv).name.name)) did not converge, aborting."
    throw(msg)
  end
  u̇_temp, = us
  _make_u!(u₊, u̇_temp, u₋, dt)

  (t₊, dt, u₊), cache
end

struct ExplicitEulerU̇Operator{N,T,U,O} <:
       AbstractSystemOperator{N,N,NonlinearOperatorType}
  t₋::T
  u₋::U
  odeop::O

  function ExplicitEulerU̇Operator(t₋, u₋, op)
    N = length(u₋)
    T = typeof(t₋)
    U = typeof(u₋)
    O = typeof(op)
    new{N,T,U,O}(t₋, u₋, op)
  end
end

function residual!(
  r::AbstractVector, op::ExplicitEulerU̇Operator,
  us::NTuple{1,AbstractVector}
)
  t₋, u₋ = op.t₋, op.u₋
  odeop = op.odeop

  k, = us

  residual!(r, odeop, t₋, u₋, k)
  r
end

function jacobian!(
  J::AbstractMatrix, op::ExplicitEulerU̇Operator, k::Val{1},
  us::NTuple{1,AbstractVector}
)
  t₋, u₋ = op.t₋, op.u₋
  odeop = op.odeop

  k, = us

  jacobian_U̇!(J, odeop, t₋, u₋, k)
  J
end

function directional_jacobian!(
  j::AbstractVector, J, op::ExplicitEulerU̇Operator, k::Val{1},
  us::NTuple{1,AbstractVector}, v::AbstractVector
)
  t₋, u₋ = op.t₋, op.u₋
  odeop = op.odeop

  k, = us

  directional_jacobian_U̇!(j, J, odeop, t₋, u₋, k, v)
  j
end

##################################
# AbstractQuasilinearODEOperator #
##################################
function allocate_subcache(
  sv::ExplicitEulerSolver, op::AbstractQuasilinearODEOperator,
  t₋::Real, dt::Real, u₋::AbstractVector,
  u̇_temp::AbstractVector, r_temp::AbstractVector, j_temp::AbstractVector
)
  T = eltype(t₋)
  J_temp = allocate_jacobian_U̇(op, T)

  fill!(u̇_temp, 0)
  jacobian_U̇!(J_temp, op, t₋, u₋, u̇_temp)
  residual!(r_temp, op, t₋, u₋, u̇_temp)
  rmul!(r_temp, -1)

  subop = LinearSystemOperator(J_temp, r_temp)
  subsv = sv.subsv
  subus = (u̇_temp,)
  subcache = allocate_cache(subsv, subop, subus)

  ExplicitEulerSolverCache(J_temp, subcache)
end

function solve!(
  u₊::AbstractVector, sv::ExplicitEulerSolver,
  op::AbstractQuasilinearODEOperator, t₋::Real, dt::Real, u₋::AbstractVector,
  cache::ODESolverCache
)
  t₊ = t₋ + dt
  u̇_temp = cache.u̇_temp
  r_temp = cache.r_temp
  J_temp = cache.subcache.J_temp

  fill!(u̇_temp, 0)
  jacobian_U̇!(J_temp, op, t₋, u₋, u̇_temp)
  residual!(r_temp, op, t₋, u₋, u̇_temp)
  rmul!(r_temp, -1)

  subop = ExplicitEulerLOperator(J_temp, r_temp)
  subsv = sv.subsv
  subcache = cache.subcache.subcache

  fill!(u̇_temp, 0)
  us = (u̇_temp,)
  us, subcache = solve!(us, subsv, subop, subcache)
  u̇_temp, = us
  _make_u!(u₊, u̇_temp, u₋, dt)

  (t₊, dt, u₊), cache
end

ExplicitEulerLOperator(A, b) = LinearSystemOperator(A, b)
