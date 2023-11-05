#################
# ExplicitEuler #
#################
struct ExplicitEuler{P,F,NLS,U̇,R,J} <: AbstractExplicitODESolver{P,F}
  problem::P
  nlsolver::NLS
  u̇_disc::U̇
  r_part::R
  j_part::J

  function ExplicitEuler(
    problem, nlsolver,
    dim::Integer, ::Type{T}, ::F
  ) where {T,F<:AbstractFormulation}
    u̇_disc = zeros(T, dim)
    r_part = zeros(T, dim)
    j_part = zeros(T, dim)

    P = typeof(problem)
    NLS = typeof(nlsolver)
    U̇ = typeof(u̇_disc)
    R = typeof(r_part)
    J = typeof(j_part)
    new{P,F,NLS,U̇,R,J}(problem, nlsolver, u̇_disc, r_part, j_part)
  end
end

_odename(::ExplicitEuler) = "Explicit Euler, explicit, order 1"
function _odedesc(::ExplicitEuler)
  """
  Euler's first-order explicit scheme.
  res(t₋, u₋, k) = 0
  """
end

function odesolve!(
  ::Type{<:AbstractODEProblem}, ::Type{<:Formulation_U},
  odesolver::ExplicitEuler, t₋, u₋, dt, u₊
)
  problem = odesolver.problem
  nlsolver = odesolver.nlsolver
  t₊ = t₋ + dt

  dt⁻¹ = inv(dt)
  u̇_disc = odesolver.u̇_disc

  function nl_res!(r, k)
    _u̇_disc!(u̇_disc, k, u₋, dt⁻¹)

    res!(problem, r, t₋, u₋, u̇_disc)
    r
  end

  function nl_jac_vec!(j, k, vec)
    _u̇_disc!(u̇_disc, k, u₋, dt⁻¹)

    jac_u̇_vec!(problem, j, t₋, u₋, u̇_disc, vec)
    rmul!(j, dt⁻¹)
    j
  end

  k = u₊
  nlsolve!(nlsolver, k, nl_res!, nl_jac_vec!)
  u₊ = k

  t₊, u₊
end

function odesolve!(
  ::Type{<:AbstractODEProblem}, ::Type{<:Formulation_U̇},
  odesolver::ExplicitEuler, t₋, u₋, dt, u₊
)
  problem = odesolver.problem
  nlsolver = odesolver.nlsolver

  t₊ = t₋ + dt

  function nl_res!(r, k)
    res!(problem, r, t₋, u₋, k)
    r
  end

  function nl_jac_vec!(j, k, vec)
    jac_u̇_vec!(problem, j, t₋, u₋, k, vec)
    j
  end

  k = u₊
  fill!(k, 0)
  nlsolve!(nlsolver, k, nl_res!, nl_jac_vec!)
  axpby!(1, u₋, dt, k)
  u₊ = k

  t₊, u₊
end

function odesolve!(
  ::Type{<:AbstractIsolatedProblem},
  odesolver::ExplicitEuler, t₋, u₋, dt, u₊
)
  problem = odesolver.problem
  lsolver = get_lsolver(problem)
  mat = get_mat(lsolver)
  vec = get_vec(lsolver)
  t₊ = t₋ + dt

  lhs!(problem, mat, t₋, u₋)
  rhs!(problem, vec, t₋, u₋)

  k = u₊
  lsolve!(lsolver, u₊)
  axpby!(1, u₋, dt, k)
  u₊ = k

  t₊, u₊
end
