##########################################
# Diagonally Implicit Runge-Kutta (DIRK) #
##########################################
struct DIRK{P,F,NLS,U̇,R,J,K,B} <: AbstractDiagonallyImplicitODESolver{P,F}
  problem::P
  nlsolver::NLS
  u̇_disc::U̇
  r_part::R
  j_part::J
  ks::K
  butcher::B

  function DIRK(
    problem, nlsolver,
    dim::Integer, ::Type{T}, ::F;
    butcher::AbstractDiagonallyImplicitButcherTableau
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

_odename(::DIRK) = "Diagonally Implicit Runge-Kutta"
function _odedesc(solver::DIRK)
  """
  Diagonally Implicit Runge-Kutta scheme corresponding to the following Butcher tableau.
  """ * tableaudisp(solver.butcher) * tableaudesc(solver.butcher)
end

function odesolve!(
  ::Type{<:AbstractODEProblem}, ::Type{<:Formulation_U},
  odesolver::DIRK, t₋, u₋, dt, u₊
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

    l = length(ais)
    for j in 1:min(l, i - 1)
      aij, kj = ais[j], ks[j]
      axpy!(aij * dt, kj, ui)
    end

    k = u₊
    copy!(k, u₋)
    if l < i || iszero(ais[i])
      function nl_res!₁(r, k)
        _u̇_disc!(u̇_disc, k, u₋, dt⁻¹)

        res!(problem, r, ti, ui, u̇_disc)
        r
      end

      function nl_jac_vec!₁(j, k, vec)
        _u̇_disc!(u̇_disc, k, u₋, dt⁻¹)

        jac_u̇_vec!(problem, j, ti, ui, u̇_disc, vec)
        rmul!(j, dt⁻¹)
        j
      end

      nlsolve!(nlsolver, k, nl_res!₁, nl_jac_vec!₁)
    else
      ai = ais[i]
      aidt = ai * dt

      function nl_res!₂(r, k)
        _u̇_disc!(u̇_disc, k, u₋, dt⁻¹)
        temp = r_part
        copy!(temp, ui)
        axpy!(aidt, u̇_disc, temp)

        res!(problem, r, ti, temp, u̇_disc)
        r
      end

      function nl_jac_vec!₂(j, k, vec)
        _u̇_disc!(u̇_disc, k, u₋, dt⁻¹)
        temp = r_part
        copy!(temp, ui)
        axpy!(aidt, u̇_disc, temp)

        jac_u̇_vec!(problem, j, ti, temp, u̇_disc, vec)
        rmul!(j, dt⁻¹)
        jac_u_vec!(problem, j_part, ti, temp, u̇_disc, vec)
        axpy!(ai, j_part, j)
        j
      end

      nlsolve!(nlsolver, k, nl_res!₂, nl_jac_vec!₂)
    end
    u₊ = k
    _u̇_disc!(ui, u₊, u₋, dt⁻¹)
  end

  # Compute us from ks
  for i in L:-1:1
    ais = as[i]
    ui = ks[i]
    l = length(ais)
    if i == L || l < i
      fill!(ui, 0)
    else
      rmul!(ui, ais[i] * dt)
    end
    axpy!(1, u₋, ui)
    for j in 1:min(l, i - 1)
      aij, kj = ais[j], ks[j]
      axpy!(aij * dt, kj, ui)
    end
  end

  # Last step
  i = L
  ais = as[i]
  l = length(ais)

  k = u₊
  copy!(k, u₋)
  if l < i || iszero(ais[i])
    function nl_res!₃(r, k)
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

    function nl_jac_vec!₃(j, k, vec)
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

    nlsolve!(nlsolver, k, nl_res!₃, nl_jac_vec!₃)
  else
    ai = ais[i]
    aidt = ai * dt

    function nl_res!₄(r, k)
      _u̇_disc!(u̇_disc, k, u₋, dt⁻¹)

      fill!(r, 0)
      for i in 1:L-1
        bi, ci = bs[i], cs[i]
        ti = t₋ + ci * dt
        ui = ks[i]
        res!(problem, r_part, ti, ui, u̇_disc)
        axpy!(bi, r_part, r)
      end
      i = L
      bi, ci = bs[i], cs[i]
      ti = t₋ + ci * dt
      ui = ks[i]
      temp = j_part
      copy!(temp, ui)
      axpy!(aidt, u̇_disc, temp)

      res!(problem, r_part, ti, temp, u̇_disc)
      axpy!(bi, r_part, r)

      r
    end

    function nl_jac_vec!₄(j, k, vec)
      _u̇_disc!(u̇_disc, k, u₋, dt⁻¹)

      fill!(j, 0)
      for i in 1:L-1
        bi, ci = bs[i], cs[i]
        ti = t₋ + ci * dt
        ui = ks[i]
        jac_u̇_vec!(problem, j_part, ti, ui, u̇_disc, vec)
        axpy!(bi * dt⁻¹, j_part, j)
      end
      i = L
      bi, ci = bs[i], cs[i]
      ti = t₋ + ci * dt
      ui = ks[i]
      temp = r_part
      copy!(temp, ui)
      axpy!(aidt, u̇_disc, temp)

      jac_u̇_vec!(problem, j_part, ti, temp, u̇_disc, vec)
      axpy!(bi * dt⁻¹, j_part, j)
      jac_u_vec!(problem, j_part, ti, temp, u̇_disc, vec)
      axpy!(bi * ai, j_part, j)

      j
    end

    nlsolve!(nlsolver, k, nl_res!₄, nl_jac_vec!₄)
  end
  u₊ = k

  t₊, u₊
end

function odesolve!(
  ::Type{<:AbstractODEProblem}, ::Type{<:Formulation_U̇},
  odesolver::DIRK, t₋, u₋, dt, u₊
)
end
