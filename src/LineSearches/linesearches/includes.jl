module LineSearches

using Optimisation.Helpers

include("Types.jl")
export StepState
export get_step
export get_value
export get_slope

export SearchState
export eval_value
export eval_slope

export get_state
export get_stepinit
export get_stepmax

export AbstractLineSearch
export list_generators
export need_slope
export search

export AbstractGenerator

export LineSearchException

include("Conditions.jl")
export AbstractCondition
export AbstractValueCondition
export AbstractSlopeCondition

export check_condition
export check_compatibility

export StrongValueCondition
export WeakValueCondition

export StrongSlopeCondition
export WeakSlopeCondition
export LightSlopeCondition

include("Interpolation.jl")
export interpolation_VS_V
export interpolation_VS0_V
export interpolation_VS_V_V
export interpolation_VS0_V_V

include("Backtracking.jl")
export Backtracking
export ExponentialDecay
export QuadraticInterpolation
export CubicInterpolation

end # module LineSearches
