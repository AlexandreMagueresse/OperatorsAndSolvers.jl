#################
# ExplicitTheta #
#################
struct ExplicitTheta{P,F,NLS,U̇,R,J,T,U0} <: AbstractExplicitODESolver{P,F}
  problem::P
  nlsolver::NLS
  u̇_disc::U̇
  r_part::R
  j_part::J
  θ::T
  u₀::U0

  function ExplicitTheta(
    problem, nlsolver,
    dim::Integer, ::Type{T}, formulation::F;
    θ=1 / 2
  ) where {T,F<:AbstractFormulation}
    msg = "θ has to be between 0 and 1."
    @assert 0 <= θ <= 1 msg
    θ = T(θ)

    if θ == 0
      ExplicitEuler(problem, nlsolver, dim, T, formulation)
    else
      u̇_disc = zeros(T, dim)
      r_part = zeros(T, dim)
      j_part = zeros(T, dim)
      u₀ = zeros(T, dim)

      P = typeof(problem)
      NLS = typeof(nlsolver)
      U̇ = typeof(u̇_disc)
      R = typeof(r_part)
      J = typeof(j_part)
      U0 = typeof(u₀)
      new{P,F,NLS,U̇,R,J,T,U0}(problem, nlsolver, u̇_disc, r_part, j_part, θ, u₀)
    end
  end
end

_odename(::ExplicitTheta) = """Explicit Theta,
* θ = 1/2 Heun, explicit, order 2
"""
function _odedesc(::ExplicitTheta)
  """
  First-order explicit Theta scheme.
  res(t₋, u₋, k) = 0
  (1 - θ) res(t₋, u₋, l) + θ res(t₊, u₋ + dt k, l) = 0
  """
end

Heun(args...; kwargs...) = ExplicitTheta(args...; θ=1 / 2, kwargs...)

function odesolve!(
  ::Type{<:AbstractODEProblem}, ::Type{<:Formulation_U},
  odesolver::ExplicitTheta, t₋, u₋, dt, u₊
)
  problem = odesolver.problem
  nlsolver = odesolver.nlsolver
  θ = odesolver.θ
  t₊ = t₋ + dt

  r_part, j_part = odesolver.r_part, odesolver.j_part

  dt⁻¹ = inv(dt)
  u̇_disc = odesolver.u̇_disc
  u₀ = odesolver.u₀

  # Explicit Euler predictor
  function nl_res!₀(r, k)
    _u̇_disc!(u̇_disc, k, u₋, dt⁻¹)

    res!(problem, r, t₋, u₋, u̇_disc)
    r
  end

  function nl_jac_vec!₀(j, k, vec)
    _u̇_disc!(u̇_disc, k, u₋, dt⁻¹)

    jac_u̇_vec!(problem, j, t₋, u₋, u̇_disc, vec)
    rmul!(j, dt⁻¹)
    j
  end

  k = u₊
  nlsolve!(nlsolver, k, nl_res!₀, nl_jac_vec!₀)
  u₊ = k
  copy!(u₀, u₊)

  # Trapezoidal corrector
  function nl_res!(r, k)
    _u̇_disc!(u̇_disc, k, u₋, dt⁻¹)

    res!(problem, r, t₋, u₋, u̇_disc)
    rmul!(r, 1 - θ)
    res!(problem, r_part, t₊, u₀, u̇_disc)
    axpy!(θ, r_part, r)
    r
  end

  function nl_jac_vec!(j, k, vec)
    _u̇_disc!(u̇_disc, k, u₋, dt⁻¹)

    jac_u̇_vec!(problem, j, t₋, u₋, u̇_disc, vec)
    rmul!(j, (1 - θ) * dt⁻¹)
    jac_u̇_vec!(problem, j_part, t₊, u₀, u̇_disc, vec)
    axpy!(θ * dt⁻¹, j_part, j)
    j
  end

  k = u₊
  nlsolve!(nlsolver, k, nl_res!, nl_jac_vec!)
  u₊ = k

  t₊, u₊
end

function odesolve!(
  ::Type{<:AbstractODEProblem}, ::Type{<:Formulation_U̇},
  odesolver::ExplicitTheta, t₋, u₋, dt, u₊
)
  problem = odesolver.problem
  nlsolver = odesolver.nlsolver
  θ = odesolver.θ
  t₊ = t₋ + dt

  r_part, j_part = odesolver.r_part, odesolver.j_part

  u₀ = odesolver.u₀

  # Explicit Euler predictor
  function nl_res!₀(r, k)
    res!(problem, r, t₋, u₋, k)
    r
  end

  function nl_jac_vec!₀(j, k, vec)
    jac_u̇_vec!(problem, j, t₋, u₋, k, vec)
    j
  end

  k = u₊
  fill!(k, 0)
  nlsolve!(nlsolver, k, nl_res!₀, nl_jac_vec!₀)
  axpby!(1, u₋, dt, k)
  u₊ = k
  copy!(u₀, u₊)

  # Trapezoidal corrector
  function nl_res!(r, k)
    res!(problem, r, t₋, u₋, k)
    rmul!(r, θ)
    res!(problem, r_part, t₊, u₀, k)
    axpy!(1 - θ, r_part, r)
    r
  end

  function nl_jac_vec!(j, k, vec)
    jac_u̇_vec!(problem, j, t₋, u₋, k, vec)
    rmul!(j, θ)
    jac_u̇_vec!(problem, j_part, t₊, u₀, k, vec)
    axpy!(1 - θ, j_part, j)
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
  odesolver::ExplicitTheta, t₋, u₋, dt, u₊
)
  problem = odesolver.problem
  θ = odesolver.θ
  t₊ = t₋ + dt

  u₀ = odesolver.u₀

  lsolver = get_lsolver(problem)
  mat = get_mat(lsolver)
  vec = get_vec(lsolver)
  mat_temp = get_mat_temp(lsolver)
  vec_temp = get_vec_temp(lsolver)

  # Explicit Euler predictor
  lhs!(problem, mat, t₋, u₋)
  rhs!(problem, vec, t₋, u₋)

  k = u₊
  lsolve!(lsolver, u₊)
  axpby!(1, u₋, dt, k)
  u₊ = k
  copy!(u₀, u₊)

  # Trapezoidal corrector
  rmul!(mat, 1 - θ)
  rmul!(vec, 1 - θ)
  lhs!(problem, mat_temp, t₊, u₀)
  rhs!(problem, vec_temp, t₊, u₀)
  axpy!(θ, mat_temp, mat)
  axpy!(θ, vec_temp, vec)

  k = u₊
  lsolve!(lsolver, u₊)
  axpby!(1, u₋, dt, k)
  u₊ = k

  t₊, u₊
end
