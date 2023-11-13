#######################################
# AbstractIterativeSystemSolverConfig #
#######################################
"""
    IterativeSystemSolverConfig

Configuration of an `IterativeSystemSolver`.
"""
struct IterativeSystemSolverConfig{I,A,R}
  maxiter::I
  atol::A
  rtol::R
end

"""
    get_maxiter(config::IterativeSystemSolverConfig) -> Integer

Return the maximum number of iterations of an iterative system solver.
"""
function get_maxiter(config::IterativeSystemSolverConfig)
  config.maxiter
end

"""
    get_convergence_state(
      config::IterativeSystemSolverConfig, r::AbstractVector
    ) -> (Bool, Number)

Compute the norm of the residual and check whether an iterative system solver
has converged absolutely.
"""
function get_convergence_state(
  config::IterativeSystemSolverConfig, r::AbstractVector
)
  atol = config.atol

  res = fastnorm(r)

  flag = NOT_CONVERGED
  if res < atol
    flag = CONVERGED_ATOL
  end

  flag, res
end

"""
    get_convergence_state(
      config::IterativeSystemSolverConfig, r::AbstractVector,
      initial_res::Number
    ) -> (Bool, number)

Compute the norm of the residual and check whether an iterative system solver
has converged absolutely or relatively.
"""
function get_convergence_state(
  config::IterativeSystemSolverConfig, r::AbstractVector,
  initial_res::Number
)
  atol, rtol = config.atol, config.rtol

  res = fastnorm(r)

  flag = NOT_CONVERGED
  if res < atol
    flag = CONVERGED_ATOL
  elseif res < rtol * initial_res
    flag = CONVERGED_RTOL
  end

  flag, res
end

"""
    fastnorm(v::AbstractVector) -> Number

Compute the Euclidean norm of a vector.
"""
function fastnorm(v::AbstractVector)
  s = zero(eltype(v))
  @inbounds @simd for vi in v
    s += abs2(vi)
  end
  sqrt(s)
end
