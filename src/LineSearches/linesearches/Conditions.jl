abstract type AbstractCondition end
abstract type AbstractValueCondition <: AbstractCondition end
abstract type AbstractSlopeCondition <: AbstractCondition end

check_condition(::AbstractCondition, ::StepState, ::StepState) = @abstractmethod

function check_compatibility(
  cond_value::AbstractValueCondition,
  cond_slope::AbstractSlopeCondition
)
  msg = "The constant of the value condition must be smaller than that of the slope condition"
  @assert cond_value.c < cond_slope.c msg
end

########################
# StrongValueCondition #
########################
# Armijo-Goldstein
struct StrongValueCondition{C} <: AbstractValueCondition
  c::C

  function StrongValueCondition(c::Real)
    msg = "Constant must lie between 0 and 1 for StrongValueCondition"
    @assert 0 < c < 1 msg
    C = typeof(c)
    new{C}(c)
  end
end

StrongValueCondition(::Type{C}) where {C} = StrongValueCondition(C(1.0e-3))

function check_condition(
  cond::StrongValueCondition,
  state0::StepState, state::StepState
)
  c = cond.c
  value0 = get_value(state0)
  slope0 = get_slope(state0)
  step = get_step(state)
  value = get_value(state)

  value <= value0 + step * c * slope0
end

######################
# WeakValueCondition #
######################
# Hager & Zhang
struct WeakValueCondition{C} <: AbstractValueCondition
  c::C

  function WeakValueCondition(c::Real)
    msg = "Constant must lie between 0 and 1 for WeakValueCondition"
    @assert 0 < c < 1 msg
    C = typeof(c)
    new{C}(2 * c - 1)
  end
end

WeakValueCondition(::Type{C}) where {C} = WeakValueCondition(C(1.0e-3))

function check_condition(
  cond::WeakValueCondition,
  state0::StepState, state::StepState
)
  c = cond.c
  slope0 = get_slope(state0)
  slope = get_slope(state)

  slope <= c * slope0
end

########################
# StrongSlopeCondition #
########################
# Strong Wolfe
struct StrongSlopeCondition{C} <: AbstractValueCondition
  c::C

  function StrongSlopeCondition(c::Real)
    msg = "Constant must lie between 0 and 1 for StrongSlopeCondition"
    @assert 0 < c < 1 msg
    C = typeof(c)
    new{C}(c)
  end
end

StrongSlopeCondition(::Type{C}) where {C} = StrongSlopeCondition(C(0.9))

function check_condition(
  cond::StrongSlopeCondition,
  state0::StepState, state::StepState
)
  c = cond.c
  slope0 = get_slope(state0)
  slope = get_slope(state)

  abs(slope) <= -c * slope0
end

######################
# WeakSlopeCondition #
######################
# Goldstein
struct WeakSlopeCondition{C} <: AbstractValueCondition
  c::C

  function WeakSlopeCondition(c::Real)
    msg = "Constant must lie between 0 and 1 for WeakSlopeCondition"
    @assert 0 < c < 1 msg
    C = typeof(c)
    new{C}(c)
  end
end

WeakSlopeCondition(::Type{C}) where {C} = WeakSlopeCondition(C(0.9))

function check_condition(
  cond::WeakSlopeCondition,
  state0::StepState, state::StepState
)
  c = cond.c
  value0 = get_value(state0)
  slope0 = get_slope(state0)
  step = get_step(state)
  value = get_value(state)

  value >= value0 + step * c * slope0
end

#######################
# LightSlopeCondition #
#######################
# Weak wolfe
struct LightSlopeCondition{C} <: AbstractValueCondition
  c::C

  function LightSlopeCondition(c::Real)
    msg = "Constant must lie between 0 and 1 for LightSlopeCondition"
    @assert 0 < c < 1 msg
    C = typeof(c)
    new{C}(c)
  end
end

LightSlopeCondition(::Type{C}) where {C} = LightSlopeCondition(C(0.9))

function check_condition(
  cond::LightSlopeCondition,
  state0::StepState, state::StepState
)
  c = cond.c
  slope0 = get_slope(state0)
  slope = get_slope(state)

  slope >= c * slope0
end
