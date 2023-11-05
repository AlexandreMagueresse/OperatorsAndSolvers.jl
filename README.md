# Solvers
Collection of solvers for ODEs, including nonlinear solvers and line search algorithms.

# TODO
* `odesolve!(::Type{<:AbstractODEProblem}, ::Type{<:Formulation_U̇}, ::DIRK)`
* Split `ButcherTableaus.jl` into separate files and have a more systematic nomenclature/names/description
  * Can probably take a look at "Diagonally Implicit Runge-Kutta, Methods for Ordinary Differential Equations. A Review" by Christopher A. Kennedy and Mark H. Carpenter
* Embedded Explicit Runge Kutta
* Embedded Diagonally Implicit Runge Kutta
* Backward Differentiation Formula
* Linear Multistep Methods
* General Linear Methods
* Bring my first attempt at line search algorithms and delete from PhD repo
* Fixed-point nonlinear solver
* More factorisations (Cholesky, QR) and more generally, allow structured Jacobians (banded, blocks)
* Counters for the number of evaluation of the residual/jacobian
* More tests
