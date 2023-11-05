struct NLConfig{L,T}
  linesearch::L
  tolerance::T
  maxiters::Int
end

####################
# AbstractNLSolver #
####################
abstract type AbstractNLSolver end

function nlsolve!(::AbstractNLSolver, args...; kwargs...)
  @abstractmethod
end

##################
# NewtonNLSolver #
##################
struct NewtonNLSolver{C,R,DU} <: AbstractNLSolver
  config::C
  r::R
  du::DU

  function NewtonNLSolver(config, dim::Integer, ::Type{T}) where {T}
    r = zeros(T, dim)
    du = zeros(T, dim)

    C = typeof(config)
    R = typeof(r)
    DU = typeof(du)
    new{C,R,DU}(config, r, du)
  end
end

function nlsolve!(solver::NewtonNLSolver, u, res!, jac_vec!)
  config = solver.config
  linesearch = config.linesearch
  tolerance = config.tolerance
  maxiters = config.maxiters

  r, du = solver.r, solver.du

  iter = 0
  while iter < maxiters
    iter += 1

    # Compute jacobian vector product
    res!(r, u)
    jac_vec!(du, u, r)
    # Compute step size
    α = lssolve(linesearch)
    # Update u
    axpy!(-α, du, u)

    # Check convergence
    # 1. Small update
    if α * norm(du) <= tolerance
      return u
    end

    # 2. Small residual
    res!(r, u)
    if norm(r) <= tolerance
      return u
    end
  end

  error("The nonlinear solver did not converge.")
end
