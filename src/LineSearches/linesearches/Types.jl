#############
# StepState #
#############
struct StepState{T,V,S}
  step::T
  value::V
  slope::S

  function StepState(step::Real, value::Real, slope::Union{Real,Nothing})
    msg = "A step size must be non-negative"
    @assert step >= 0 msg

    T = typeof(step)
    V = typeof(value)
    S = typeof(slope)
    new{T,V,S}(step, value, slope)
  end
end

get_step(as::StepState) = as.step
get_value(as::StepState) = as.value
get_slope(as::StepState) = as.slope

###############
# SearchState #
###############
struct SearchState{F,D,S,SI,SM}
  fun::F
  der::D
  state::S
  stepinit::SI
  stepmax::SM

  function SearchState(
    fun::Function, der::Function,
    value::Real, slope::Real,
    stepinit::Real, stepmax::Real
  )
    msg = "This research direction is not a descent direction"
    @assert slope < 0 msg

    F = typeof(fun)
    D = typeof(der)

    SI = typeof(stepinit)
    SM = typeof(stepmax)
    T = promote_type(SI, SM)

    state = StepState(T(0), value, slope)
    S = typeof(state)

    new{F,D,S,SI,SM}(fun, der, state, stepinit, stepmax)
  end
end

eval_value(ss::SearchState, step::Real) = ss.fun(step)
eval_slope(ss::SearchState, step::Real) = ss.der(step)

get_state(ss::SearchState) = ss.state
get_stepinit(ss::SearchState) = ss.stepinit
get_stepmax(ss::SearchState) = ss.stepmax

######################
# AbstractLineSearch #
######################
abstract type AbstractLineSearch end

list_generators(::Type{<:AbstractLineSearch}) = @abstractmethod
need_slope(::AbstractLineSearch) = @abstractmethod
search(::AbstractLineSearch, ::SearchState) = @abstractmethod

function StepState(step::Real, ss::SearchState, ls::AbstractLineSearch)
  T = typeof(step)
  value = eval_value(ss, step)::T

  slope = nothing
  if need_slope(ls)
    slope = eval_slope(ss, step)
  end

  StepState(step, value, slope)
end

#####################
# AbstractGenerator #
#####################
abstract type AbstractGenerator end

need_slope(::AbstractGenerator) = @abstractmethod

#######################
# LineSearchException #
#######################
struct LineSearchException{M,S} <: Exception
  msg::M
  state::S
end
