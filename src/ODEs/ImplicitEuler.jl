#######################
# ImplicitEulerSolver #
#######################
"""
    ImplicitEulerSolver

Euler's Implicit scheme.
Type          Implicit
Order         1
Description   res(t₊, u₊, k) = 0
"""
struct ImplicitEulerSolver{F,S} <:
       AbstractODESolver{F,ImplicitODESolverType}
  subsv::S

  function ImplicitEulerSolver{F}(subsv) where {F}
    S = typeof(subsv)
    new{F,S}(subsv)
  end
end

ImplicitEulerSolver(subsv) = ImplicitEulerSolver{UFormulationType}(subsv)

"""
    ImplicitEulerSolverCache

Cache corresponding to an `ImplicitEulerSolver`.
"""
struct ImplicitEulerSolverCache{JM,JM2,C} <:
       AbstractSolverCache
  J_temp::JM
  J_temp2::JM2
  subcache::C
end

####################
# UFormulationType #
####################
function allocate_subcache(
  T::AbstractOperatorType, F::UFormulationType,
  sv::ImplicitEulerSolver, op::AbstractODEOperator,
  t₋::Real, dt::Real, u₋::AbstractVector,
  u̇_temp::AbstractVector, r_temp::AbstractVector, j_temp::AbstractVector
)
  T = eltype(t₋)
  J_temp = allocate_jacobian_U̇(op, T)

  t₊ = t₋ + dt
  dt⁻¹ = inv(dt)

  subop = ImplicitEulerUOperator(t₊, u₋, dt⁻¹, u̇_temp, j_temp, J_temp, op)
  subsv = sv.subsv
  subus = (r_temp,)
  subcache = allocate_cache(subsv, subop, subus)

  ImplicitEulerSolverCache(J_temp, nothing, subcache)
end

function solve!(
  u₊::AbstractVector, T::AbstractOperatorType, F::UFormulationType,
  sv::ImplicitEulerSolver, op::AbstractODEOperator,
  t₋::Real, dt::Real, u₋::AbstractVector,
  cache::ODESolverCache
)
  t₊ = t₋ + dt
  dt⁻¹ = inv(dt)
  u̇_temp = cache.u̇_temp
  j_temp = cache.j_temp
  J_temp = cache.subcache.J_temp

  subop = ImplicitEulerUOperator(t₊, u₋, dt⁻¹, u̇_temp, j_temp, J_temp, op)
  subsv = sv.subsv
  subcache = cache.subcache.subcache

  us = (u₊,)
  if subsv isa AbstractIterativeSystemSolver
    reset_cache!(subcache, subsv, subop, us)
  end
  us, subcache = solve!(us, subsv, subop, subcache)
  if subcache isa IterativeSystemSolverCache && !isconverged(subcache)
    msg = "$(string(typeof(subsv).name.name)) did not converge, aborting."
    throw(msg)
  end
  u₊, = us

  (t₊, dt, u₊), cache
end

struct ImplicitEulerUOperator{N,T,U,D,U̇,JV,JM,O} <:
       AbstractSystemOperator{N,N,NonlinearOperatorType}
  t₊::T
  u₋::U
  dt⁻¹::D
  u̇_temp::U̇
  j_temp::JV
  J_temp::JM
  odeop::O

  function ImplicitEulerUOperator(t₊, u₋, dt⁻¹, u̇_temp, j_temp, J_temp, op)
    N = length(u₋)
    T = typeof(t₊)
    U = typeof(u₋)
    D = typeof(dt⁻¹)
    U̇ = typeof(u̇_temp)
    JV = typeof(j_temp)
    JM = typeof(J_temp)
    O = typeof(op)
    new{N,T,U,D,U̇,JV,JM,O}(t₊, u₋, dt⁻¹, u̇_temp, j_temp, J_temp, op)
  end
end

function residual!(
  r::AbstractVector, op::ImplicitEulerUOperator,
  us::NTuple{1,AbstractVector}
)
  t₊, u₋, dt⁻¹, u̇_temp = op.t₊, op.u₋, op.dt⁻¹, op.u̇_temp
  odeop = op.odeop

  u, = us
  _make_u̇!(u̇_temp, u, u₋, dt⁻¹)

  residual!(r, odeop, t₊, u, u̇_temp)
  r
end

function jacobian!(
  J::AbstractMatrix, op::ImplicitEulerUOperator, k::Val{1},
  us::NTuple{1,AbstractVector}
)
  t₊, u₋, dt⁻¹, u̇_temp, J_temp = op.t₊, op.u₋, op.dt⁻¹, op.u̇_temp, op.J_temp
  odeop = op.odeop

  u, = us
  _make_u̇!(u̇_temp, u, u₋, dt⁻¹)

  jacobian_U!(J, odeop, t₊, u, u̇_temp)
  jacobian_U̇!(J_temp, odeop, t₊, u, u̇_temp)
  axpy!(dt⁻¹, J_temp, J)
  J
end

function directional_jacobian!(
  j::AbstractVector, J, op::ImplicitEulerUOperator, k::Val{1},
  us::NTuple{1,AbstractVector}, v::AbstractVector
)
  t₊, u₋, dt⁻¹, u̇_temp, j_temp = op.t₊, op.u₋, op.dt⁻¹, op.u̇_temp, op.j_temp
  odeop = op.odeop

  u, = us
  _make_u̇!(u̇_temp, u, u₋, dt⁻¹)

  directional_jacobian_U!(j, J, odeop, t₊, u, u̇_temp, v)
  directional_jacobian_U̇!(j_temp, J, odeop, t₊, u, u̇_temp, v)
  axpy!(dt⁻¹, j_temp, j)
  j
end

####################
# U̇FormulationType #
####################
function allocate_subcache(
  T::AbstractOperatorType, F::U̇FormulationType,
  sv::ImplicitEulerSolver, op::AbstractODEOperator,
  t₋::Real, dt::Real, u₋::AbstractVector,
  u̇_temp::AbstractVector, r_temp::AbstractVector, j_temp::AbstractVector
)
  T = eltype(t₋)
  J_temp = allocate_jacobian_U̇(op, T)

  t₊ = t₋ + dt
  subop = ImplicitEulerU̇Operator(t₊, u₋, dt, j_temp, J_temp, op)
  subsv = sv.subsv
  subus = (u̇_temp,)
  subcache = allocate_cache(subsv, subop, subus)

  ImplicitEulerSolverCache(J_temp, nothing, subcache)
end

function solve!(
  u₊::AbstractVector, T::AbstractOperatorType, F::U̇FormulationType,
  sv::ImplicitEulerSolver, op::AbstractODEOperator,
  t₋::Real, dt::Real, u₋::AbstractVector,
  cache::ODESolverCache
)
  t₊ = t₋ + dt
  u̇_temp = cache.u̇_temp
  j_temp = cache.j_temp
  J_temp = cache.subcache.J_temp

  subop = ImplicitEulerU̇Operator(t₊, u₋, dt, j_temp, J_temp, op)
  subsv = sv.subsv
  subcache = cache.subcache.subcache

  fill!(u̇_temp, 0)
  us = (u̇_temp,)
  if subsv isa AbstractIterativeSystemSolver
    reset_cache!(subcache, subsv, subop, us)
  end
  us, subcache = solve!(us, subsv, subop, subcache)
  if subcache isa IterativeSystemSolverCache && !isconverged(subcache)
    msg = "$(string(typeof(subsv).name.name)) did not converge, aborting."
    throw(msg)
  end
  u̇_temp, = us
  _make_u!(u₊, u̇_temp, u₋, dt)

  (t₊, dt, u₊), cache
end

struct ImplicitEulerU̇Operator{N,T,U,D,JV,JM,O} <:
       AbstractSystemOperator{N,N,NonlinearOperatorType}
  t₊::T
  u₋::U
  dt::D
  j_temp::JV
  J_temp::JM
  odeop::O

  function ImplicitEulerU̇Operator(t₊, u₋, dt, j_temp, J_temp, op)
    N = length(u₋)
    T = typeof(t₊)
    U = typeof(u₋)
    D = typeof(dt)
    JV = typeof(j_temp)
    JM = typeof(J_temp)
    O = typeof(op)
    new{N,T,U,D,JV,JM,O}(t₊, u₋, dt, j_temp, J_temp, op)
  end
end

function residual!(
  r::AbstractVector, op::ImplicitEulerU̇Operator,
  us::NTuple{1,AbstractVector}
)
  t₊, dt, u₋, j_temp = op.t₊, op.dt, op.u₋, op.j_temp
  odeop = op.odeop

  k, = us

  u_temp = u₋
  axpy!(dt, k, u_temp)
  residual!(r, odeop, t₊, u_temp, k)
  axpy!(-dt, k, u_temp)
  r
end

function jacobian!(
  J::AbstractMatrix, op::ImplicitEulerU̇Operator, k::Val{1},
  us::NTuple{1,AbstractVector}
)
  t₊, dt, u₋, J_temp = op.t₊, op.dt, op.u₋, op.J_temp
  odeop = op.odeop

  k, = us

  u_temp = u₋
  axpy!(dt, k, u_temp)
  jacobian_U̇!(J, odeop, t₊, u_temp, k)
  jacobian_U!(J_temp, odeop, t₊, u_temp, k)
  axpy!(dt, J_temp, J)
  axpy!(-dt, k, u_temp)
  J
end

function directional_jacobian!(
  j::AbstractVector, J, op::ImplicitEulerU̇Operator, k::Val{1},
  us::NTuple{1,AbstractVector}, v::AbstractVector
)
  t₊, dt, u₋, j_temp = op.t₊, op.dt, op.u₋, op.j_temp
  odeop = op.odeop

  k, = us

  u_temp = u₋
  axpy!(dt, k, u_temp)
  directional_jacobian_U̇!(j, J, odeop, t₊, u_temp, k, v)
  directional_jacobian_U!(j_temp, J, odeop, t₊, u_temp, k, v)
  axpy!(dt, j_temp, j)
  axpy!(-dt, k, u_temp)
  j
end

##############################
# AbstractLinearOperatorType #
##############################
function allocate_subcache(
  T::AbstractLinearOperatorType,
  sv::ImplicitEulerSolver, op::AbstractODEOperator,
  t₋::Real, dt::Real, u₋::AbstractVector,
  u̇_temp::AbstractVector, r_temp::AbstractVector, j_temp::AbstractVector
)
  T = eltype(t₋)
  J_temp = allocate_jacobian_U̇(op, T)
  J_temp2 = allocate_jacobian_U̇(op, T)

  fill!(u̇_temp, 0)
  jacobian_U̇!(J_temp, op, t₋, u₋, u̇_temp)
  residual!(r_temp, op, t₋, u₋, u̇_temp)
  rmul!(r_temp, -1)

  subop = LinearSystemOperator(J_temp, r_temp)
  subsv = sv.subsv
  subus = (u̇_temp,)
  subcache = allocate_cache(subsv, subop, subus)

  ImplicitEulerSolverCache(J_temp, J_temp2, subcache)
end

function solve!(
  u₊::AbstractVector, T::AbstractLinearOperatorType,
  sv::ImplicitEulerSolver, op::AbstractODEOperator,
  t₋::Real, dt::Real, u₋::AbstractVector,
  cache::ODESolverCache
)
  t₊ = t₋ + dt
  u̇_temp = cache.u̇_temp
  r_temp = cache.r_temp
  J_temp = cache.subcache.J_temp
  J_temp2 = cache.subcache.J_temp2

  # Some tricks here
  # M u̇ = K u + f
  # M u̇ = K (u₋ + dt u̇) + f
  # (M - dt K) u̇ = K u₋ + f
  # M - dt K = jac_U̇(u=Any,u̇=Any) + dt * jac_U(u=Any,u̇=Any)
  # K u₋ + f = -res(U=u₋,U̇=0)
  fill!(u̇_temp, 0)
  jacobian_U̇!(J_temp, op, t₊, u₋, u̇_temp)
  jacobian_U!(J_temp2, op, t₊, u₋, u̇_temp)
  axpy!(dt, J_temp2, J_temp)
  residual!(r_temp, op, t₊, u₋, u̇_temp)
  rmul!(r_temp, -1)

  subop = ImplicitEulerLOperator(J_temp, r_temp)
  subsv = sv.subsv
  subcache = cache.subcache.subcache

  us = (u̇_temp,)
  if subsv isa AbstractIterativeSystemSolver
    reset_cache!(subcache, subsv, subop, us)
  end
  us, subcache = solve!(us, subsv, subop, subcache)
  if subcache isa IterativeSystemSolverCache && !isconverged(subcache)
    msg = "$(string(typeof(subsv).name.name)) did not converge, aborting."
    throw(msg)
  end
  u̇_temp, = us
  _make_u!(u₊, u̇_temp, u₋, dt)

  (t₊, dt, u₊), cache
end

ImplicitEulerLOperator(A, b) = LinearSystemOperator(A, b)
