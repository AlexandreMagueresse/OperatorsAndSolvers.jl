###################
# ConstantStepper #
###################
"""
    ConstantStepper

Line search solver that outputs a constant step size, i.e. α(k) = α(0).
"""
struct ConstantStepper{T} <:
       AbstractLineSearchSolver
  step::T
end

"""
    ConstantStepperCache

Cache corresponding to `ConstantStepper`.
"""
struct ConstantStepperCache <:
       AbstractSolverCache end

# AbstractSolver interface
function allocate_initial_guess(
  sv::ConstantStepper, op::AbstractLineSearchOperator, ::Type{T}
) where {T}
  T[sv.step]
end

function allocate_cache(
  sv::ConstantStepper, op::AbstractLineSearchOperator,
  us::NTuple{1,AbstractVector}
)
  ConstantStepperCache()
end

function solve!(
  us::NTuple{1,AbstractVector}, ls::ConstantStepper,
  op::AbstractLineSearchOperator, cache::ConstantStepperCache
)
  u, = us
  u[1] = ls.step

  us = (u,)
  (us, cache)
end

######################
# ExponentialStepper #
######################
"""
    ExponentialStepper

Line search solver that multiplies the initial step by a constant factor every
time it is called, i.e. α(k) = γ^k * α(0).
"""
struct ExponentialStepper{A,Γ} <:
       AbstractLineSearchSolver
  α0::A
  γ::Γ
end

"""
    ExponentialStepperCache

Cache corresponding to `ExponentialStepper`.
"""
struct ExponentialStepperCache{S} <:
       AbstractSolverCache
  step::S
end

# AbstractSolver interface
function allocate_initial_guess(
  sv::ExponentialStepper, op::AbstractLineSearchOperator, ::Type{T}
) where {T}
  T[sv.α0]
end

function allocate_cache(
  sv::ExponentialStepper, op::AbstractLineSearchOperator,
  us::NTuple{1,AbstractVector}
)
  ExponentialStepperCache(Ref(sv.α0))
end

function solve!(
  us::NTuple{1,AbstractVector}, ls::ExponentialStepper,
  op::AbstractLineSearchOperator, cache::ExponentialStepperCache
)
  u, = us
  α = cache.step[]
  u[1] = α

  cache.step[] *= ls.γ
  us = (u,)
  ((u,), cache)
end
