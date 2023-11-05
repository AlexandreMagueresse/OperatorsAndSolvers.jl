#################
# ImplicitEuler #
#################
struct ImplicitEuler{P,F,NLS,U̇,R,J} <: AbstractDiagonallyImplicitODESolver{P,F}
  problem::P
  nlsolver::NLS
  u̇_disc::U̇
  r_part::R
  j_part::J

  function ImplicitEuler(
    problem, nlsolver,
    dim::Integer, ::Type{T}, ::F
  ) where {T,F<:AbstractFormulation}
    u̇_disc = zeros(T, dim)
    r_part = zeros(T, dim)
    j_part = zeros(T, dim)

    P = typeof(problem)
    NLS = typeof(nlsolver)
    U̇ = typeof(u̇_disc)
    R = typeof(j_part)
    J = typeof(j_part)
    new{P,F,NLS,U̇,R,J}(problem, nlsolver, u̇_disc, r_part, j_part)
  end
end

_odename(::ImplicitEuler) = "Implicit Euler, implicit, order 1"
function _odedesc(::ImplicitEuler)
  """
  Euler's first-order implicit scheme.
  res(t₊, u₋ + dt * k, k) = 0
  """
end

function odesolve!(
  ::Type{<:AbstractODEProblem}, ::Type{<:Formulation_U},
  odesolver::ImplicitEuler, t₋, u₋, dt, u₊
)
  problem = odesolver.problem
  nlsolver = odesolver.nlsolver
  t₊ = t₋ + dt

  j_part = odesolver.j_part

  dt⁻¹ = inv(dt)
  u̇_disc = odesolver.u̇_disc

  function nl_res!(r, k)
    _u̇_disc!(u̇_disc, k, u₋, dt⁻¹)

    res!(problem, r, t₊, k, u̇_disc)
    r
  end

  function nl_jac_vec!(j, k, vec)
    _u̇_disc!(u̇_disc, k, u₋, dt⁻¹)

    jac_u_vec!(problem, j, t₊, k, u̇_disc, vec)
    jac_u̇_vec!(problem, j_part, t₊, k, u̇_disc, vec)
    axpy!(dt⁻¹, j_part, j)
    j
  end

  k = u₊
  nlsolve!(nlsolver, k, nl_res!, nl_jac_vec!)
  u₊ = k

  t₊, u₊
end

function odesolve!(
  ::Type{<:AbstractODEProblem}, ::Type{<:Formulation_U̇},
  odesolver::ImplicitEuler, t₋, u₋, dt, u₊
)
  problem = odesolver.problem
  nlsolver = odesolver.nlsolver
  t₊ = t₋ + dt

  r_part, j_part = odesolver.r_part, odesolver.j_part

  function nl_res!(r, k)
    temp = j_part
    copy!(temp, u₋)
    axpy!(dt, k, temp)

    res!(problem, r, t₊, temp, k)
    r
  end

  function nl_jac_vec!(j, k, vec)
    temp = r_part
    copy!(temp, u₋)
    axpy!(dt, k, temp)

    jac_u̇_vec!(problem, j, t₊, temp, k, vec)
    jac_u_vec!(problem, j_part, t₊, temp, k, vec)
    axpy!(dt, j_part, j)
    j
  end

  k = u₊
  fill!(k, 0)
  nlsolve!(nlsolver, k, nl_res!, nl_jac_vec!)
  axpby!(1, u₋, dt, k)
  u₊ = k

  t₊, u₊
end
