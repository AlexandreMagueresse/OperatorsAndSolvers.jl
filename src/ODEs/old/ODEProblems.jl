abstract type AbstractODEProblem{T,D} end

function res!(::AbstractODEProblem, v, t, u, u̇)
  @abstractmethod
end

function jac_u_vec!(::AbstractODEProblem, v, t, u, u̇, vec)
  @abstractmethod
end

function jac_u̇_vec!(::AbstractODEProblem, v, t, u, u̇, vec)
  @abstractmethod
end

#####################
# GenericODEProblem #
#####################
# ODE of the form res(t, u, u̇) = 0
struct GenericODEProblem{T,D,R,JUV,JU̇V} <: AbstractODEProblem{T,D}
  res!::R
  jac_u_vec!::JUV
  jac_u̇_vec!::JU̇V

  function GenericODEProblem(
    res!, jac_u_vec!, jac_u̇_vec!,
    dim::Integer, ::Type{T}
  ) where {T}
    D = dim

    R = typeof(res!)
    JUV = typeof(jac_u_vec!)
    JU̇V = typeof(jac_u̇_vec!)
    new{T,D,R,JUV,JU̇V}(res!, jac_u_vec!, jac_u̇_vec!)
  end
end

# AbstractODEProblem interface
function res!(problem::GenericODEProblem, v, t, u, u̇)
  problem.res!(v, t, u, u̇)
  v
end

function jac_u_vec!(problem::GenericODEProblem, v, t, u, u̇, vec)
  problem.jac_u_vec!(v, t, u, u̇, vec)
  v
end

function jac_u̇_vec!(problem::GenericODEProblem, v, t, u, u̇, vec)
  problem.jac_u̇_vec!(v, t, u, u̇, vec)
  v
end

###########################
# AbstractIsolatedProblem #
###########################
abstract type AbstractIsolatedProblem{T,D} <: AbstractODEProblem{T,D} end

function lhs!(::AbstractIsolatedProblem, M, t, u)
  @abstractmethod
end

function rhs!(::AbstractIsolatedProblem, v, t, u)
  @abstractmethod
end

function get_lsolver(::AbstractIsolatedProblem)
  @abstractmethod
end

######################
# IsolatedODEProblem #
######################
# ODE of the form lhs(t, u) u̇ = rhs(t, u)
# lhs!(M, t, u)
# lhs_vec!(v, t, u, vec)
# rhs!(v, t, u)
# jac_lhs_vec!(v, t, u, u̇, vec)
# jac_rhs_vec!(v, t, u, vec)
struct IsolatedODEProblem{T,D,L,LV,R,JLV,JRV,LS,V,M} <: AbstractIsolatedProblem{T,D}
  lhs!::L
  lhs_vec!::LV
  rhs!::R
  jac_lhs_vec!::JLV
  jac_rhs_vec!::JRV
  lsolver::LS
  v_temp::V
  M_temp::M

  function IsolatedODEProblem(
    lhs!, lhs_vec!, rhs!, jac_lhs_vec!, jac_rhs_vec!, lsolver,
    dim::Integer, ::Type{T}
  ) where {T}
    v_temp = zeros(T, (dim,))
    M_temp = zeros(T, (dim, dim))
    D = dim

    L = typeof(lhs!)
    LV = typeof(lhs_vec!)
    R = typeof(rhs!)
    JLV = typeof(jac_lhs_vec!)
    JRV = typeof(jac_rhs_vec!)
    LS = typeof(lsolver)
    V = typeof(v_temp)
    M = typeof(M_temp)
    new{T,D,L,LV,R,JLV,JRV,LS,V,M}(
      lhs!, lhs_vec!, rhs!,
      jac_lhs_vec!, jac_rhs_vec!,
      lsolver, v_temp, M_temp
    )
  end
end

# AbstractODEProblem interface
function _lhs_vec!(problem::IsolatedODEProblem, v, t, u, vec)
  if !isnothing(problem.lhs_vec!)
    problem.lhs_vec!(v, t, u, vec)
  else
    M_temp = problem.M_temp
    problem.lhs!(M_temp, t, u)
    mul!(v, M_temp, vec)
  end
  v
end

function res!(problem::IsolatedODEProblem, v, t, u, u̇)
  # v = lhs(t, u) * u̇ - rhs(t, u)
  v_temp = problem.v_temp

  _lhs_vec!(problem, v, t, u, u̇)
  problem.rhs!(v_temp, t, u)
  axpy!(-1, v_temp, v)
  v
end

function jac_u_vec!(problem::IsolatedODEProblem, v, t, u, u̇, vec)
  # jac_lhs(t, u) * u̇ - jac_rhs(t, u)
  v_temp = problem.v_temp

  problem.jac_lhs_vec!(v, t, u, u̇, vec)
  problem.jac_rhs_vec!(v_temp, t, u, vec)
  axpy!(-1, v_temp, v)
  v
end

function jac_u̇_vec!(problem::IsolatedODEProblem, v, t, u, u̇, vec)
  _lhs_vec!(problem, v, t, u, vec)
  v
end

# AbstractIsolatedProblem interface
function lhs!(problem::IsolatedODEProblem, M, t, u)
  problem.lhs!(M, t, u)
end

function rhs!(problem::IsolatedODEProblem, v, t, u)
  problem.rhs!(v, t, u)
end

function get_lsolver(problem::IsolatedODEProblem)
  problem.lsolver
end

########################
# LinearisedODEProblem #
########################
# ODE obtained by linearising a GenericODEProblem
# res(t, u, u̇) = 0 <=> jac_u̇(t, u, u̇₀) (u̇ - u̇₀) + res(t, u, u̇₀) = 0
struct LinearisedODEProblem{T,D,L,R,RE,JUV,JU̇V,LS} <: AbstractIsolatedProblem{T,D}
  lhs!::L
  rhs!::R
  res!::RE
  jac_u_vec!::JUV
  jac_u̇_vec!::JU̇V
  lsolver::LS

  function LinearisedODEProblem(
    lhs!, rhs!, res!, jac_u_vec!, jac_u̇_vec!, lsolver,
    dim::Integer, ::Type{T}
  ) where {T}
    D = dim

    L = typeof(lhs!)
    R = typeof(rhs!)
    RE = typeof(res!)
    JUV = typeof(jac_u_vec!)
    JU̇V = typeof(jac_u̇_vec!)
    LS = typeof(lsolver)
    new{T,D,L,R,RE,JUV,JU̇V,LS}(
      lhs!, rhs!, res!,
      jac_u_vec!, jac_u̇_vec!,
      lsolver
    )
  end
end

# AbstractODEProblem interface
function res!(problem::LinearisedODEProblem, v, t, u, u̇)
  problem.res!(v, t, u, u̇)
  v
end

function jac_u_vec!(problem::LinearisedODEProblem, v, t, u, u̇, vec)
  problem.jac_u_vec!(v, t, u, u̇, vec)
  v
end

function jac_u̇_vec!(problem::LinearisedODEProblem, v, t, u, u̇, vec)
  problem.jac_u̇_vec!(v, t, u, vec)
  v
end

# AbstractIsolatedProblem interface
function lhs!(problem::LinearisedODEProblem, M, t, u)
  problem.lhs!(M, t, u)
  M
end

function rhs!(problem::LinearisedODEProblem, M, t, u)
  problem.rhs!(M, t, u)
  M
end

function get_lsolver(problem::LinearisedODEProblem)
  problem.lsolver
end

###############
# Conversions #
###############
function to_generic(problem::IsolatedODEProblem{T,D}) where {T,D}
  # rhs(t, u) * u̇ - lhs(t, u)
  dim = D

  _res!(v, t, u, u̇) = res!(problem, v, t, u, u̇)
  _jac_u_vec!(v, t, u, u̇, vec) = jac_u_vec!(problem, j, t, u, u̇, vec)
  _jac_u̇_vec!(v, t, u, u̇, vec) = jac_u̇_vec!(problem, j, t, u, u̇, vec)
  GenericODEProblem(_res!, _jac_u_vec!, _jac_u̇_vec!, dim, T)
end

function to_isolated(problem::GenericODEProblem{T,D}, u̇₀, jac_u_jac_u̇_vec) where {T,D}
  # jac_u̇(t, u, u̇₀) (u̇ - u̇₀) = f(t, u, u₀)
  dim = D
  v_temp = zeros(T, (dim,))

  _lhs!(M, t, u) = jac_u̇!(problem, M, t, u, u̇₀)
  function _rhs!(v, t, u)
    jac_u̇_vec!(problem, v, t, u, u̇₀, u̇₀)
    res!(problem, v_temp, t, u, u̇₀)
    axpy!(-1, v_temp, v)
    v
  end
  function _res!(v, t, u, u̇)
    copy!(v, u̇)
    axpy!(-1, u̇₀, v)
    jac_u̇_vec!(problem, v_temp, t, u, u̇₀, v)
    copy!(v, v_temp)
    res!(problem, v_temp, t, u, u̇₀)
    axpy!(+1, v_temp, v)
    v
  end
  function _jac_u_vec!(v, t, u, u̇, vec)
    copy!(v, u̇)
    axpy!(-1, u̇₀, v)
    jac_u_jac_u̇_vec(v_temp, t, u, u̇₀, v)
    copy!(v, v_temp)
    jac_u_vec!(problem, v_temp, t, u, u̇₀, vec)
    axpy!(+1, v_temp, v)
    v
  end
  _jac_u̇_vec!(v, t, u, vec) = jac_u̇_vec!(problem, v, t, u, u̇₀, vec)

  lsolver = LULSolver(dim, T)
  LinearisedODEProblem(
    _lhs!, _res!, _rhs!,
    _jac_u_vec!, _jac_u̇_vec!,
    dim, T, lsolver
  )
end
