using OperatorsAndSolvers
import OperatorsAndSolvers.AbstractTypes: residual!
import OperatorsAndSolvers.ODEs: jacobian_U!
import OperatorsAndSolvers.ODEs: jacobian_U̇!
import OperatorsAndSolvers.ODEs: directional_jacobian_U!
import OperatorsAndSolvers.ODEs: directional_jacobian_U̇!

using LinearAlgebra
using Plots

##############
# ODEProblem #
##############
T = Float64

# problem_id = 1 # Stiff linear ODE
# problem_id = 2 # Nonlinear ODE
# problem_id = 3 # Harmonic oscillator
problem_id = 4 # 1D minimisation (Gradient flow)

if problem_id == 1
  # u̇ = λ * (u - g(t)) + g'(t)
  N = 1

  struct MyOp1{T,F,G} <: AbstractODEOperator{N,NonlinearOperatorType}
    λ::T
    g::F
    ∇g::G
  end

  function residual!(
    r::AbstractVector, op::MyOp1,
    t::Real, u::AbstractVector, u̇::AbstractVector
  )
    gt, ∇gt = op.g(t), op.∇g(t)
    r[1] = u̇[1] - op.λ * (u[1] - gt) - ∇gt
    r
  end

  function jacobian_U!(
    J::AbstractMatrix, op::MyOp1,
    t::Real, u::AbstractVector, u̇::AbstractVector
  )
    J[1, 1] = -op.λ
    J
  end

  function jacobian_U̇!(
    J::AbstractMatrix, op::MyOp1,
    t::Real, u::AbstractVector, u̇::AbstractVector
  )
    J[1, 1] = 1
    J
  end

  function directional_jacobian_U!(
    j::AbstractVector, J, op::MyOp1,
    t::Real, u::AbstractVector, u̇::AbstractVector, v::AbstractVector
  )
    j[1] = -op.λ * v[1]
    j
  end

  function directional_jacobian_U̇!(
    j::AbstractVector, J, op::MyOp1,
    t::Real, u::AbstractVector, u̇::AbstractVector, v::AbstractVector
  )
    j[1] = v[1]
    j
  end

  ######################
  # Initial conditions #
  ######################
  λ = T(-1000)
  ω = T(10)
  g(t) = cospi(ω * t)
  ∇g(t) = -ω * sinpi(ω * t) * pi
  op = MyOp1(λ, g, ∇g)
  t₋, tₑ = T(0), T(0.05)
  u₋ = T[0]

  has_solution = true
  A = u₋[1] - g(t₋)
  function solution!(u::AbstractVector, t::Real)
    gt = g(t)
    expt = exp(λ * t)
    u[1] = A * expt + gt
    u
  end
elseif problem_id == 2
  # u̇ = 1 - u^2
  N = 1

  struct MyOp2 <: AbstractODEOperator{N,NonlinearOperatorType}
  end

  function residual!(
    r::AbstractVector, op::MyOp2,
    t::Real, u::AbstractVector, u̇::AbstractVector
  )
    r[1] = u̇[1] - 1 + u[1]^2
    r
  end

  function jacobian_U!(
    J::AbstractMatrix, op::MyOp2,
    t::Real, u::AbstractVector, u̇::AbstractVector
  )
    J[1, 1] = 2 * u[1]
    J
  end

  function jacobian_U̇!(
    J::AbstractMatrix, op::MyOp2,
    t::Real, u::AbstractVector, u̇::AbstractVector
  )
    J[1, 1] = 1
    J
  end

  function directional_jacobian_U!(
    j::AbstractVector, J, op::MyOp2,
    t::Real, u::AbstractVector, u̇::AbstractVector, v::AbstractVector
  )
    j[1] = 2 * u[1] * v[1]
    j
  end

  function directional_jacobian_U̇!(
    j::AbstractVector, J, op::MyOp2,
    t::Real, u::AbstractVector, u̇::AbstractVector, v::AbstractVector
  )
    j[1] = v[1]
    j
  end

  ######################
  # Initial conditions #
  ######################
  op = MyOp2()
  t₋, tₑ = T(0), T(10.0)
  u₋ = T[0]

  has_solution = true
  A = u₋[1]
  function solution!(u::AbstractVector, t::Real)
    tanht = tanh(t)
    u[1] = A + tanht
    u
  end
elseif problem_id == 3
  # ü + ω² u = 0
  # dU̇ = [0 1, -ω² 0] U (u̇,-ω²u)
  N = 2

  struct MyOp3{T} <: AbstractODEOperator{N,NonlinearOperatorType}
    ω²::T
  end

  function residual!(
    r::AbstractVector, op::MyOp3,
    t::Real, u::AbstractVector, u̇::AbstractVector
  )
    r[1] = u̇[1] - u[2]
    r[2] = u̇[2] + op.ω² * u[1]
    r
  end

  function jacobian_U!(
    J::AbstractMatrix, op::MyOp3,
    t::Real, u::AbstractVector, u̇::AbstractVector
  )
    J[1, 1] = 0
    J[1, 2] = -1
    J[2, 1] = op.ω²
    J[2, 2] = 0
    J
  end

  function jacobian_U̇!(
    J::AbstractMatrix, op::MyOp3,
    t::Real, u::AbstractVector, u̇::AbstractVector
  )
    J[1, 1] = 1
    J[1, 2] = 0
    J[2, 1] = 0
    J[2, 2] = 1
    J
  end

  function directional_jacobian_U!(
    j::AbstractVector, J, op::MyOp3,
    t::Real, u::AbstractVector, u̇::AbstractVector, v::AbstractVector
  )
    j[1] = -v[2]
    j[2] = op.ω² * v[1]
    j
  end

  function directional_jacobian_U̇!(
    j::AbstractVector, J, op::MyOp3,
    t::Real, u::AbstractVector, u̇::AbstractVector, v::AbstractVector
  )
    j[1] = v[1]
    j[2] = v[2]
    j
  end

  ######################
  # Initial conditions #
  ######################
  ω = T(5)
  ω² = ω^2
  B = T(3)
  op = MyOp3(ω²)
  t₋, tₑ = T(0), T(3.0)
  u₋ = T[0, B*ω]

  has_solution = true
  A = u₋[1] - g(t₋)
  function solution!(u::AbstractVector, t::Real)
    u[1] = B * sin(ω * t)
    u[2] = B * ω * cos(ω * t)
    u
  end
elseif problem_id == 4
  # u̇ = -∇f(u)
  N = 1

  struct MyOp4{F,G} <: AbstractODEOperator{N,NonlinearOperatorType}
    ∇f::F
    ∇²f::G
  end

  function residual!(
    r::AbstractVector, op::MyOp4,
    t::Real, u::AbstractVector, u̇::AbstractVector
  )
    ∇fu = op.∇f(u[1])
    r[1] = u̇[1] + ∇fu
    r
  end

  function jacobian_U!(
    J::AbstractMatrix, op::MyOp4,
    t::Real, u::AbstractVector, u̇::AbstractVector
  )
    ∇²fu = op.∇²f(u[1])
    J[1, 1] = ∇²fu
    J
  end

  function jacobian_U̇!(
    J::AbstractMatrix, op::MyOp4,
    t::Real, u::AbstractVector, u̇::AbstractVector
  )
    J[1, 1] = 1
    J
  end

  function directional_jacobian_U!(
    j::AbstractVector, J, op::MyOp4,
    t::Real, u::AbstractVector, u̇::AbstractVector, v::AbstractVector
  )
    ∇²fu = op.∇²f(u[1])
    j[1] = ∇²fu * v[1]
    j
  end

  function directional_jacobian_U̇!(
    j::AbstractVector, J, op::MyOp4,
    t::Real, u::AbstractVector, u̇::AbstractVector, v::AbstractVector
  )
    j[1] = v[1]
    j
  end

  ######################
  # Initial conditions #
  ######################
  f(x) = exp(x) - cos(x)
  ∇f(x) = exp(x) + sin(x)
  ∇²f(x) = exp(x) + cos(x)

  op = MyOp4(∇f, ∇²f)
  t₋, tₑ = T(0), T(10)
  u₋ = T[-3.1]

  has_solution = false
end

dt = (tₑ - t₋) / 100

#################
# System solver #
#################
maxiter = 1_000
atol = 1000 * eps(T)
rtol = 1000 * eps(T)
config = IterativeSystemSolverConfig(maxiter, atol, rtol)

F = Formulation_U
# F = Formulation_U̇

lsv = LUSolver()
subsv = NewtonRaphsonSolver(lsv, config)

if F == Formulation_U
  α = T(dt^2 / 3)
elseif F == Formulation_U̇
  α = T(1 / 3)
end
lssv = ConstantStepper(α)
subsv = GradientDescentSolver(lssv, config)

##############
# ODE Solver #
##############
sv = ExplicitEulerSolver{F}(subsv)

#############
# Main loop #
#############
u₊ = copy(u₋)
uₛ = similar(u₋)
cache = allocate_cache(sv, op, t₋, dt, u₋)

plot()
c = 1
sim_start = time_ns()
while t₋ < tₑ
  scatter!([t₋], [u₋[c]], ms=2, ma=1.0, mc="blue", label="")
  if has_solution
    solution!(uₛ, t₋)
    scatter!([t₋], [uₛ[c]], ms=2, ma=0.5, mc="green", label="")
  end

  if t₋ + dt > tₑ
    break
  end
  (t₊, dt, u₊), cache = solve!(u₊, sv, op, t₋, dt, u₋)
  t₋ = t₊
  copy!(u₋, u₊)
end
sim_end = time_ns()
sim_elapsed = (sim_end - sim_start) / 1e9

scatter!([t₋], [u₋[c]], ms=2, ma=1.0, mc="blue", label="Estimate")
if has_solution
  solution!(uₛ, t₋)
  scatter!([t₋], [uₛ[c]], ms=2, ma=0.5, mc="green", label="Reference")
end
display(plot!())

println(sim_elapsed)
