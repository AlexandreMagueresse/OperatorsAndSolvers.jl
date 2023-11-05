using Solvers

using LinearAlgebra
using Plots

###############
# ODEProblem #
###############
T = Float64

problem_id = 1 # Stiff linear ODE
problem_id = 2 # Nonlinear ODE
problem_id = 3 # Harmonic oscillator
problem_id = 4 # 1D minimisation (Gradient flow)

if problem_id == 1
  dim = 1

  # Residual and jacobians
  # u̇ = λ * (u - g(t)) + g'(t)

  # Generic description
  λ = T(-1000)
  ω = T(10)
  function g!(v, t)
    v[1] = cospi(ω * t)
  end
  function jac_g!(v, t)
    v[1] = -ω * sinpi(ω * t) * pi
  end

  w = zeros(T, dim)
  function res!(v, t, u, u̇)
    copy!(v, u)
    g!(w, t)
    axpy!(-1, w, v)
    rmul!(v, λ)
    jac_g!(w, t)
    axpy!(+1, w, v)
    axpy!(-1, u̇, v)
    v
  end

  function jac_u_vec!(v, t, u, u̇, vec)
    copy!(v, vec)
    rmul!(v, λ)
    v
  end

  function jac_u̇_vec!(v, t, u, u̇, vec)
    copy!(v, vec)
    rmul!(v, -1)
    v
  end

  # Isolated description
  is_isolated = true

  function lhs!(M, t, u)
    copy!(M, I(dim))
    M
  end

  function lhs_vec!(v, t, u, vec)
    copy!(v, vec)
    v
  end

  function rhs!(v, t, u)
    copy!(v, u)
    g!(w, t)
    axpy!(-1, w, v)
    rmul!(v, λ)
    jac_g!(w, t)
    axpy!(+1, w, v)
    v
  end

  function jac_lhs_vec!(v, t, u, u̇, vec)
    fill!(v, 0)
    v
  end

  function jac_rhs_vec!(v, t, u, vec)
    copy!(v, vec)
    rmul!(v, λ)
    v
  end

  # Initial conditions
  tstart, tend, dt = T(0), T(0.05), T(0.001)
  ustart = T[0]

  # Solution
  has_solution = true
  wstart = zeros(T, dim)
  g!(wstart, T(0))
  wstart .= ustart .- wstart

  function sol!(v, t)
    expt = exp(λ * t)
    for i in 1:dim
      v[i] = wstart[i] * expt
    end
    g!(w, t)
    for i in 1:dim
      v[i] += w[i]
    end
    v
  end
elseif problem_id == 2
  dim = 1

  # Residual and jacobians
  # u̇ = 1 - u^2

  # Generic description
  function res!(v, t, u, u̇)
    @inbounds for i in 1:dim
      v[i] = 1 - u[i]^2 - u̇[i]
    end
  end

  function jac_u_vec!(v, t, u, u̇, vec)
    @inbounds for i in 1:dim
      v[i] = -2 * u[i] * vec[i]
    end
    v
  end

  function jac_u̇_vec!(v, t, u, u̇, vec)
    copy!(v, vec)
    rmul!(v, -1)
    v
  end

  # Isolated description
  is_isolated = true

  function lhs!(M, t, u)
    copy!(M, I(dim))
  end

  function lhs_vec!(v, t, u, vec)
    copy!(v, vec)
    v
  end

  function rhs!(v, t, u)
    @inbounds for i in 1:dim
      v[i] = 1 - u[i]^2
    end
    v
  end

  function jac_lhs_vec!(v, t, u, u̇, vec)
    fill!(v, 0)
    v
  end

  function jac_rhs_vec!(v, t, u, vec)
    @inbounds for i in 1:dim
      v[i] = -2 * u[i] * vec[i]
    end
    v
  end

  # Initial conditions
  tstart, tend, dt = T(0), T(10), T(0.1)
  ustart = T[0]

  # Solution
  has_solution = true
  function sol!(v, t)
    tanht = tanh(t)
    for i in 1:dim
      v[i] = tanht + ustart[i]
    end
    v
  end
elseif problem_id == 3
  dim = 2

  # Residual and jacobians
  # ü + ω² u = 0
  ω = 5
  ω² = ω^2

  # Generic description
  function res!(v, t, u, u̇)
    v[1] = u[2] - u̇[1]
    v[2] = -ω² * u[1] - u̇[2]
    v
  end

  function jac_u_vec!(v, t, u, u̇, vec)
    v[1] = vec[2]
    v[2] = -ω² * vec[1]
    v
  end

  function jac_u̇_vec!(v, t, u, u̇, vec)
    copy!(v, vec)
    rmul!(v, -1)
    v
  end

  # Isolated description
  is_isolated = true

  function lhs!(M, t, u)
    copy!(M, I(dim))
  end

  function lhs_vec!(v, t, u, vec)
    copy!(v, vec)
    v
  end

  function rhs!(v, t, u)
    v[1] = u[2]
    v[2] = -ω² * u[1]
    v
  end

  function jac_lhs_vec!(v, t, u, u̇, vec)
    fill!(v, 0)
    v
  end

  function jac_rhs_vec!(v, t, u, vec)
    v[1] = vec[2]
    v[2] = -ω² * vec[1]
    v
  end

  # Initial conditions
  B = 3
  tstart, tend, dt = T(0), T(10), T(0.05)
  ustart = T[0, B*ω]

  # Solution
  has_solution = true
  function sol!(v, t)
    v[1] = B * sin(ω * t)
    v[2] = B * ω * cos(ω * t)
    v
  end
elseif problem_id == 4
  dim = 1

  # Residual and jacobians
  # u̇ = -∇f(u)
  f(x) = exp(x) - cos(x)
  ∇f(x) = exp(x) + sin(x)
  ∇²f(x) = exp(x) + cos(x)

  # Generic description
  function res!(v, t, u, u̇)
    v[1] = -∇f(u[1]) - u̇[1]
    v
  end

  function jac_u_vec!(v, t, u, u̇, vec)
    v[1] = -∇²f(u[1]) * vec[1]
    v
  end

  function jac_u̇_vec!(v, t, u, u̇, vec)
    copy!(v, vec)
    rmul!(v, -1)
    v
  end

  # Isolated description
  is_isolated = true

  function lhs!(M, t, u)
    copy!(M, I(dim))
  end

  function lhs_vec!(v, t, u, vec)
    copy!(v, vec)
    v
  end

  function rhs!(v, t, u)
    v[1] = -∇f(u[1])
    v
  end

  function jac_lhs_vec!(v, t, u, u̇, vec)
    fill!(v, 0)
    v
  end

  function jac_rhs_vec!(v, t, u, vec)
    v[1] = -∇²f(u[1]) * vec[1]
    v
  end

  # Initial conditions
  tstart, tend, dt = T(0), T(10), T(1)
  ustart = T[-3.1]

  has_solution = false
end

if is_isolated
  lsolver = BackslashLSolver(dim, T)
  # lsolver = LULSolver(dim, T)
  problem = IsolatedODEProblem(
    lhs!, lhs_vec!, rhs!,
    jac_lhs_vec!, jac_rhs_vec!,
    lsolver, dim, T
  )
else
  problem = GenericODEProblem(
    res!,
    jac_u_vec!, jac_u̇_vec!,
    dim, T
  )
end

############
# NLSolver #
############
formulation = Formulation_U()
# formulation = Formulation_U̇()

if formulation isa Formulation_U
  α = T(dt^2 / 3)
elseif formulation isa Formulation_U̇
  α = T(1 / 3)
end
lssolver = ConstantLSSolver(α)

tol = 1000 * eps(T)
maxiters = 1000
nlconfig = NLConfig(lssolver, tol, maxiters)

nlsolver = NewtonNLSolver(nlconfig, dim, T)

#############
# ODESolver #
#############
# odesolver = ExplicitEuler(problem, nlsolver, dim, T, formulation)
# odesolver = ImplicitEuler(problem, nlsolver, dim, T, formulation)

# odesolver = CrankNicolson(problem, nlsolver, dim, T, formulation)
# θ = T(1 / 2)
# odesolver = ImplicitTheta(problem, nlsolver, dim, T, formulation; θ)

# odesolver = Heun(problem, nlsolver, dim, T, formulation)
# θ = T(1 / 2)
# odesolver = ExplicitTheta(problem, nlsolver, dim, T, formulation; θ)

butcher = DIB_4_4_CeschinoKunzmann(T)
odesolver = RungeKutta(problem, nlsolver, dim, T, formulation; butcher)

#############
# Main loop #
#############
t = tstart
uest, unew = copy(ustart), similar(ustart)
uref = similar(ustart)
k = 0

plot()
c = 1
sim_start = time_ns()
# while t < tend
for _ in 1:100
  scatter!([t], [uest[c]], ms=2, ma=1.0, mc="blue", label="")
  if has_solution
    sol!(uref, t)
    scatter!([t], [uref[c]], ms=2, ma=0.5, mc="green", label="")
  end

  if t + dt > tend
    break
  end
  t, unew = odesolve!(odesolver, t, uest, dt, unew)
  copy!(uest, unew)
  k += 1

  # if mod(k, 10) == 0
  #   display(plot!())
  # end
end
sim_end = time_ns()
sim_elapsed = (sim_end - sim_start) / 1e9

scatter!([t], [uest[c]], ms=2, ma=1.0, mc="blue", label="Estimate")
if has_solution
  sol!(uref, t)
  scatter!([t], [uref[c]], ms=2, ma=0.5, mc="green", label="Reference")
end
display(plot!())

println(sim_elapsed)
