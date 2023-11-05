#######################
# AbstractFormulation #
#######################
abstract type AbstractFormulation end

struct Formulation_U <: AbstractFormulation
end

struct Formulation_U̇ <: AbstractFormulation
end

#####################
# AbstractODESolver #
#####################
abstract type AbstractODESolver{P,F} end
abstract type AbstractExplicitODESolver{P,F} <: AbstractODESolver{P,F} end
abstract type AbstractImplicitODESolver{P,F} <: AbstractODESolver{P,F} end
abstract type AbstractDiagonallyImplicitODESolver{P,F} <: AbstractImplicitODESolver{P,F} end

function odename(solver::AbstractODESolver)
  println(_odename(solver))
end
_odename(::AbstractODESolver) = ""

function odedesc(solver::AbstractODESolver)
  println(_odedesc(solver))
end
_odedesc(::AbstractODESolver) = ""

function odesolve!(
  odesolver::AbstractODESolver{P},
  args...; kwargs...
) where {P}
  odesolve!(P, odesolver, args...; kwargs...)
end

function odesolve!(
  ::Type{<:AbstractODEProblem},
  odesolver::AbstractODESolver{P,F},
  args...; kwargs...
) where {P,F}
  odesolve!(P, F, odesolver, args...; kwargs...)
end

include("ODESolvers/ExplicitEuler.jl")
include("ODESolvers/ImplicitEuler.jl")
include("ODESolvers/ExplicitTheta.jl")
include("ODESolvers/ImplicitTheta.jl")
include("ODESolvers/ERK.jl")
include("ODESolvers/DIRK.jl")
include("ODESolvers/RungeKutta.jl")
