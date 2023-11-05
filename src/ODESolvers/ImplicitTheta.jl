#################
# ImplicitTheta #
#################
struct ImplicitTheta{P,F,NLS,U̇,R,J,T} <: AbstractDiagonallyImplicitODESolver{P,F}
  problem::P
  nlsolver::NLS
  u̇_disc::U̇
  r_part::R
  j_part::J
  θ::T

  function ImplicitTheta(
    problem, nlsolver,
    dim::Integer, ::Type{T}, formulation::F;
    θ=1 / 2
  ) where {T,F<:AbstractFormulation}
    msg = "θ has to be between 0 and 1."
    @assert 0 <= θ <= 1 msg
    θ = T(θ)

    if θ == 0
      ExplicitEuler(problem, nlsolver, dim, T, formulation)
    elseif θ == 1
      ImplicitEuler(problem, nlsolver, dim, T, formulation)
    else
      u̇_disc = zeros(T, dim)
      r_part = zeros(T, dim)
      j_part = zeros(T, dim)

      P = typeof(problem)
      NLS = typeof(nlsolver)
      U̇ = typeof(u̇_disc)
      R = typeof(r_part)
      J = typeof(j_part)
      new{P,F,NLS,U̇,R,J,T}(problem, nlsolver, u̇_disc, r_part, j_part, θ)
    end
  end
end

_odename(::ImplicitTheta) = """Implicit Theta,
* θ = 0    Implicit Euler, implicit, order 1
* θ = 1/2  Crank-Nicolson, implicit, order 2
* θ = 1    Explicit Euler, explicit, order 1
"""
function _odedesc(solver::ImplicitTheta)
  """
  First-order implicit Theta scheme.
  (1 - θ) res(t₋, u₋, k) + θ res(t₊, u₋ + dt k, k) = 0
  """
end

CrankNicolson(args...; kwargs...) = ImplicitTheta(args...; θ=1 / 2, kwargs...)

function odesolve!(
  ::Type{<:AbstractODEProblem}, ::Type{<:Formulation_U},
  odesolver::ImplicitTheta, t₋, u₋, dt, u₊
)
  problem = odesolver.problem
  nlsolver = odesolver.nlsolver
  θ = odesolver.θ
  t₊ = t₋ + dt

  r_part, j_part = odesolver.r_part, odesolver.j_part

  dt⁻¹ = inv(dt)
  u̇_disc = odesolver.u̇_disc

  function nl_res!(r, k)
    _u̇_disc!(u̇_disc, k, u₋, dt⁻¹)

    res!(problem, r, t₋, u₋, u̇_disc)
    rmul!(r, 1 - θ)
    res!(problem, r_part, t₊, k, u̇_disc)
    axpy!(θ, r_part, r)
    r
  end

  function nl_jac_vec!(j, k, vec)
    _u̇_disc!(u̇_disc, k, u₋, dt⁻¹)

    jac_u̇_vec!(problem, j, t₋, u₋, u̇_disc, vec)
    rmul!(j, (1 - θ) * dt⁻¹)
    jac_u_vec!(problem, j_part, t₊, k, u̇_disc, vec)
    axpy!(θ, j_part, j)
    jac_u̇_vec!(problem, j_part, t₊, k, u̇_disc, vec)
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
  odesolver::ImplicitTheta, t₋, u₋, dt, u₊
)
  problem = odesolver.problem
  nlsolver = odesolver.nlsolver
  θ = odesolver.θ
  t₊ = t₋ + dt

  r_part, j_part = odesolver.r_part, odesolver.j_part

  function nl_res!(r, k)
    temp = j_part
    copy!(temp, u₋)
    axpy!(dt, k, temp)

    res!(problem, r, t₋, u₋, k)
    rmul!(r, 1 - θ)
    res!(problem, r_part, t₊, temp, k)
    axpy!(θ, r_part, r)
    r
  end

  function nl_jac_vec!(j, k, vec)
    temp = r_part
    copy!(temp, u₋)
    axpy!(dt, k, temp)

    jac_u̇_vec!(problem, j, t₋, u₋, k, vec)
    rmul!(j, 1 - θ)
    jac_u_vec!(problem, j_part, t₊, temp, k, vec)
    axpy!(θ * dt, j_part, j)
    jac_u̇_vec!(problem, j_part, t₊, temp, k, vec)
    axpy!(θ, j_part, j)
    j
  end

  k = u₊
  fill!(k, 0)
  nlsolve!(nlsolver, k, nl_res!, nl_jac_vec!)
  axpby!(1, u₋, dt, k)
  u₊ = k

  t₊, u₊
end
