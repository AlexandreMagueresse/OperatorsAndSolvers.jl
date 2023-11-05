####################
# AbstractLSSolver #
####################
abstract type AbstractLSSolver end

function lssolve(::AbstractLSSolver, args...; kwargs...)
@abstractmethod
end

####################
# ConstantLSSolver #
####################
struct ConstantLSSolver{T} <: AbstractLSSolver
  step::T
end

function lssolve(ls::ConstantLSSolver, args...; kwargs...)
  ls.step
end
