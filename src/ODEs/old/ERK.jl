##############################
# Explicit Runge-Kutta (ERK) #
##############################
struct ERK{P,F,NLS,U̇,R,J,K,B} <: AbstractExplicitODESolver{P,F}
  problem::P
  nlsolver::NLS
  u̇_disc::U̇
  r_part::R
  j_part::J
  ks::K
  butcher::B

  function ERK(
    problem, nlsolver,
    dim::Integer, ::Type{T}, ::F;
    butcher::AbstractExplicitButcherTableau
  ) where {T,F}
    u̇_disc = zeros(T, dim)
    r_part = zeros(T, dim)
    j_part = zeros(T, dim)

    L = length(butcher)
    ks = [zeros(T, dim) for _ in 1:L]

    P = typeof(problem)
    NLS = typeof(nlsolver)
    U̇ = typeof(u̇_disc)
    R = typeof(r_part)
    J = typeof(j_part)
    K = typeof(ks)
    B = typeof(butcher)
    new{P,F,NLS,U̇,R,J,K,B}(
      problem, nlsolver,
      u̇_disc, r_part, j_part,
      ks, butcher
    )
  end
end

_odename(::ERK) = "Explicit Runge-Kutta"
function _odedesc(solver::ERK)
  """
  Explicit Runge-Kutta scheme corresponding to the following Butcher tableau.
  """ * tableaudisp(solver.butcher) * tableaudesc(solver.butcher)
end

function odesolve!(
  ::Type{<:AbstractODEProblem}, ::Type{<:Formulation_U},
  odesolver::ERK, t₋, u₋, dt, u₊
)
  problem = odesolver.problem
  nlsolver = odesolver.nlsolver
  t₊ = t₋ + dt

  r_part, j_part = odesolver.r_part, odesolver.j_part

  dt⁻¹ = inv(dt)
  u̇_disc = odesolver.u̇_disc

  ks = odesolver.ks
  butcher = odesolver.butcher
  as, bs, cs = butcher.as, butcher.bs, butcher.cs
  L = length(butcher)

  # Intermediate steps except last
  for i in 1:L-1
    ais, ci = as[i], cs[i]
    ti = t₋ + ci * dt
    ui = ks[i]
    copy!(ui, u₋)
    for (aij, kj) in zip(ais, ks)
      axpy!(aij * dt, kj, ui)
    end

    function nl_res!(r, k)
      _u̇_disc!(u̇_disc, k, u₋, dt⁻¹)

      res!(problem, r, ti, ui, u̇_disc)
      r
    end

    function nl_jac_vec!(j, k, vec)
      _u̇_disc!(u̇_disc, k, u₋, dt⁻¹)

      jac_u̇_vec!(problem, j, ti, ui, u̇_disc, vec)
      rmul!(j, dt⁻¹)
      j
    end

    k = u₊
    copy!(k, u₋)
    nlsolve!(nlsolver, k, nl_res!, nl_jac_vec!)
    u₊ = k
    _u̇_disc!(ui, u₊, u₋, dt⁻¹)
  end

  # Compute last us from ks
  for i in L:-1:1
    ais = as[i]
    ui = ks[i]
    copy!(ui, u₋)
    for (aij, kj) in zip(ais, ks)
      axpy!(aij * dt, kj, ui)
    end
  end

  # Last step
  function nl_res!ₑ(r, k)
    _u̇_disc!(u̇_disc, k, u₋, dt⁻¹)

    fill!(r, 0)
    for i in 1:L
      bi, ci = bs[i], cs[i]
      ti = t₋ + ci * dt
      ui = ks[i]
      res!(problem, r_part, ti, ui, u̇_disc)
      axpy!(bi, r_part, r)
    end
    r
  end

  function nl_jac_vec!ₑ(j, k, vec)
    _u̇_disc!(u̇_disc, k, u₋, dt⁻¹)

    fill!(j, 0)
    for i in 1:L
      bi, ci = bs[i], cs[i]
      ti = t₋ + ci * dt
      ui = ks[i]
      jac_u̇_vec!(problem, j_part, ti, ui, u̇_disc, vec)
      axpy!(bi * dt⁻¹, j_part, j)
    end
    j
  end

  k = u₊
  copy!(k, u₋)
  nlsolve!(nlsolver, k, nl_res!ₑ, nl_jac_vec!ₑ)
  u₊ = k

  t₊, u₊
end

function odesolve!(
  ::Type{<:AbstractODEProblem}, ::Type{<:Formulation_U̇},
  odesolver::ERK, t₋, u₋, dt, u₊
)
  problem = odesolver.problem
  nlsolver = odesolver.nlsolver
  t₊ = t₋ + dt

  r_part, j_part = odesolver.r_part, odesolver.j_part

  ks = odesolver.ks
  butcher = odesolver.butcher
  as, bs, cs = butcher.as, butcher.bs, butcher.cs
  L = length(butcher)

  # Intermediate steps except last
  for i in 1:L-1
    ais, ci = as[i], cs[i]
    ti = t₋ + ci * dt
    ui = ks[i]
    copy!(ui, u₋)
    for (aij, kj) in zip(ais, ks)
      axpy!(aij * dt, kj, ui)
    end

    function nl_res!(r, k)
      res!(problem, r, ti, ui, k)
      r
    end

    function nl_jac_vec!(j, k, vec)
      jac_u̇_vec!(problem, j, ti, ui, k, vec)
      j
    end

    k = u₊
    copy!(k, u₋)
    nlsolve!(nlsolver, k, nl_res!, nl_jac_vec!)
    copy!(ks[i], k)
  end

  # Compute us from ks
  for i in L:-1:1
    ais = as[i]
    ui = ks[i]
    copy!(ui, u₋)
    for (aij, kj) in zip(ais, ks)
      axpy!(aij * dt, kj, ui)
    end
  end

  # Last step
  function nl_res!ₑ(r, k)
    fill!(r, 0)
    for i in 1:L
      bi, ci = bs[i], cs[i]
      ti = t₋ + ci * dt
      ui = ks[i]
      res!(problem, r_part, ti, ui, k)
      axpy!(bi, r_part, r)
    end
    r
  end

  function nl_jac_vec!ₑ(j, k, vec)
    fill!(j, 0)
    for i in 1:L
      bi, ci = bs[i], cs[i]
      ti = t₋ + ci * dt
      ui = ks[i]
      jac_u̇_vec!(problem, j_part, ti, ui, k, vec)
      axpy!(bi, j_part, j)
    end
    j
  end

  k = u₊
  fill!(k, 0)
  nlsolve!(nlsolver, k, nl_res!ₑ, nl_jac_vec!ₑ)
  axpby!(1, u₋, dt, k)
  u₊ = k

  t₊, u₊
end

function odesolve!(
  ::Type{<:AbstractIsolatedProblem},
  odesolver::ERK, t₋, u₋, dt, u₊
)
  problem = odesolver.problem
  lsolver = get_lsolver(problem)
  mat = get_mat(lsolver)
  vec = get_vec(lsolver)
  mat_temp = get_mat_temp(lsolver)
  vec_temp = get_vec_temp(lsolver)
  t₊ = t₋ + dt

  ks = odesolver.ks
  butcher = odesolver.butcher
  as, bs, cs = butcher.as, butcher.bs, butcher.cs
  L = length(butcher)

  # Intermediate steps except last
  fill!(mat_temp, 0)
  fill!(vec_temp, 0)
  for i in 1:L-1
    ais, bi, ci = as[i], bs[i], cs[i]
    ti = t₋ + ci * dt
    ui = ks[i]
    copy!(ui, u₋)
    for (aij, kj) in zip(ais, ks)
      axpy!(aij * dt, kj, ui)
    end

    lhs!(problem, mat, ti, ui)
    rhs!(problem, vec, ti, ui)
    axpy!(bi, mat, mat_temp)
    axpy!(bi, vec, vec_temp)
    lsolve!(lsolver, ks[i])
  end

  # Last step
  i = L
  ais, bi, ci = as[i], bs[i], cs[i]
  ti = t₋ + ci * dt
  ui = ks[i]
  copy!(ui, u₋)
  for (aij, kj) in zip(ais, ks)
    axpy!(aij * dt, kj, ui)
  end

  lhs!(problem, mat, ti, ui)
  rhs!(problem, vec, ti, ui)
  axpy!(bi, mat, mat_temp)
  axpy!(bi, vec, vec_temp)

  copy!(mat, mat_temp)
  copy!(vec, vec_temp)

  k = u₊
  lsolve!(lsolver, u₊)
  axpby!(1, u₋, dt, k)
  u₊ = k

  t₊, u₊
end
