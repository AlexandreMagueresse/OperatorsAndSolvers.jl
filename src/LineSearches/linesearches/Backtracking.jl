################
# Backtracking #
################
struct Backtracking{C,G,T} <: AbstractLineSearch
  cond_value::C
  gen::G

  ρ_min::T
  ρ_max::T
  iter_max::Int

  function Backtracking(
    cond_value::AbstractValueCondition, ::Type{T};
    ρ_min::Real=T(0.1), ρ_max::Real=T(0.9), iter_max::Int=1000,
    gen::AbstractGenerator=CubicInterpolation()
  ) where {T}
    msg = "Minimum decay rate must lie between 0 and 1"
    @assert 0 < ρ_min < 1 msg
    msg = "Maximum decay rate must lie between 0 and 1"
    @assert 0 < ρ_max < 1 msg
    msg = "Number of maximum iterations must be positive"
    @assert iter_max > 0

    if ρ_min > ρ_max
      ρ_min, ρ_max = ρ_max, ρ_min
    end

    C = typeof(cond_value)
    G = typeof(gen)
    new{C,G,T}(
      cond_value, gen,
      ρ_min, ρ_max, iter_max
    )
  end
end

need_slope(ls::Backtracking) = need_slope(ls.gen)

list_gens(::Type{Backtracking}) = (
  :ExponentialDecay,
  :QuadraticInterpolation,
  :CubicInterpolation,
)

function search(ls::Backtracking, ss::SearchState)
  cond_value = ls.cond_value
  gen = ls.gen
  ρ_min = ls.ρ_min
  ρ_max = ls.ρ_max

  state0 = get_state(ss)

  stepinit = get_stepinit(ss)
  stepmax = get_stepmax(ss)
  stepinit = min(stepinit, stepmax)
  state = StepState(stepinit, ss, ls)

  reset_generator!(gen)
  iter = -1
  while !check_condition(cond_value, state0, state)
    iter += 1
    if iter > ls.iter_max
      msg = "Failed to converge after $(iter) iteration(s)"
      throw(LineSearchException(msg, state))
    end

    # Generate new step
    new_step = generate(gen, state0, state)

    # Ensure reasonable decrease of the step
    old_step = get_step(state)
    new_step = clamp(new_step, ρ_min * old_step, ρ_max * old_step)

    # Move to next state
    state = StepState(new_step, ss, ls)
  end
  state
end

####################
# ExponentialDecay #
####################
struct ExponentialDecay{T} <: AbstractGenerator
  ρ::T

  function ExponentialDecay(ρ::Real)
    msg = "The decay rate must lie strictly between 0 and 1"
    @assert 0 < ρ < 1 msg

    T = typeof(ρ)
    new{T}(ρ)
  end
end

need_slope(::ExponentialDecay) = false
reset_generator!(::ExponentialDecay) = nothing

function generate(
  gen::ExponentialDecay,
  ::StepState, state::StepState
)
  gen.ρ * get_step(state)
end

##########################
# QuadraticInterpolation #
##########################
struct QuadraticInterpolation <: AbstractGenerator
end

need_slope(::QuadraticInterpolation) = false
reset_generator!(::QuadraticInterpolation) = nothing

function generate(
  ::QuadraticInterpolation,
  state0::StepState, state::StepState
)
  value0 = get_value(state0)
  slope0 = get_slope(state0)

  step1 = get_step(state)
  value1 = get_value(state)

  interpolation_VS0_V(
    value0, slope0,
    step1, value1
  )
end

######################
# CubicInterpolation #
######################
mutable struct CubicInterpolation <: AbstractGenerator
  prevstate

  function CubicInterpolation()
    prevstate = nothing
    new(prevstate)
  end
end

need_slope(::CubicInterpolation) = false
reset_generator!(gen::CubicInterpolation) = gen.prevstate = nothing

function generate(
  gen::CubicInterpolation,
  state0::StepState, state::StepState
)
  value0 = get_value(state0)
  slope0 = get_slope(state0)

  step1 = get_step(state)
  value1 = get_value(state)
  T = typeof(value1)

  prevstate = gen.prevstate
  step2 = nothing
  value2 = nothing
  if !isnothing(prevstate)
    step2 = get_step(prevstate)::T
    value2 = get_value(prevstate)::T
  end

  # Initialise with quadratic interp.
  # Ensure that previous value > current value
  # So that the cubic interp. has a minimum
  # Otherwise default to quadratic interp.

  if isnothing(value2) || (value2 < value1)
    new_step = interpolation_VS0_V(
      value0, slope0,
      step1, value1
    )
  else
    step_quad = interpolation_VS0_V(
      value0, slope0,
      step1, value1
    )

    step_cub = interpolation_VS0_V_V(
      value0, slope0,
      step1, value1,
      step2, value2,
    )

    new_step = step_cub
    if isapprox(new_step, 0)
      new_step = step_quad
    end
  end

  gen.prevstate = state
  new_step
end
