module Solvers

using LinearAlgebra

include("Utils.jl")

include("LSolvers.jl")
export AbstractLSolver
export get_mat
export get_vec
export get_mat_temp
export get_vec_temp
export lsolve!

export BackslashLSolver
export LULSolver

include("LSSolvers.jl")
export AbstractLSSolver
export lssolve

export ConstantLSSolver

include("NLSolvers.jl")
export NLConfig

export AbstractNLSolver
export nlsolve!

export NewtonNLSolver

include("ButcherTableaus.jl")
export AbstractButcherTableau
export AbstractExplicitButcherTableau
export AbstractImplicitButcherTableau
export AbstractDiagonallyImplicitButcherTableau
export tableaudisp
export tableauname
export tableaudesc

export ExplicitButcherTableau

export EB_1_1
export EB_1_1_Euler

export EB_2_2
export EB_2_2_Midpoint
export EB_2_2_Ralston
export EB_2_2_Heun

export EB_3_3
export EB_3_3_Kutta
export EB_3_3_Ralston
export EB_3_3_Heun
export EB_3_3_Wray
export EB_3_3_Houwen
export EB_3_3_SSPRK

export EB_4_4_Kutta
export EB_4_4_Simpson
export EB_4_4_Ralston

export DiagonallyImplicitButcherTableau

export DIB_1_1
export DIB_1_1_Euler

export DIB_2_1_MidPoint
export DIB_2_2_CrankNicolson
export DIB_2_2_PareschiRusso
export DIB_2_2_QinZhang
export DIB_2_2_Unknown
export DIB_3_2_Crouzeix
export DIB_3_2_HammerHollingsworth
export DIB_3_3_CeschinoKunzmann
export DIB_3_3_Alt

export DIB_4_3_Crouzeix
export DIB_4_3_Butcher
export DIB_4_4_CeschinoKunzmann
export DIB_4_4_Alt

include("ODEProblems.jl")
export AbstractODEProblem
export res!
export jac_u_vec!
export jac_u̇_vec!

export GenericODEProblem

export AbstractIsolatedProblem
export lhs!
export rhs!
export get_lsolver

export IsolatedODEProblem
export LinearisedODEProblem

export to_generic
export to_isolated

include("ODESolvers.jl")
export AbstractFormulation
export Formulation_U
export Formulation_U̇

export AbstractODESolver
export AbstractExplicitODESolver
export AbstractImplicitODESolver
export AbstractDiagonallyImplicitODESolver
export odename
export odedesc
export odesolve!

export ExplicitEuler
export ImplicitEuler

export ImplicitTheta
export CrankNicolson

export ExplicitTheta
export Heun

export ERK
# export DIRK
# export SDIRK
export RungeKutta

end # module Solvers
