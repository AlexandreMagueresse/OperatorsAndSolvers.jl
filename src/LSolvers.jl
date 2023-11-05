###################
# AbstractLSolver #
###################
abstract type AbstractLSolver end

function get_mat(::AbstractLSolver)
  @abstractmethod
end

function get_vec(::AbstractLSolver)
  @abstractmethod
end

function get_mat_temp(::AbstractLSolver)
  @abstractmethod
end

function get_vec_temp(::AbstractLSolver)
  @abstractmethod
end

function lsolve!(::AbstractLSolver, v)
  @abstractmethod
end

####################
# BackslashLSolver #
####################
struct BackslashLSolver{M,V,MT,VT} <: AbstractLSolver
  mat::M
  vec::V
  mat_temp::MT
  vec_temp::VT

  function BackslashLSolver(dim::Integer, ::Type{T}) where {T}
    mat = zeros(T, (dim, dim))
    vec = zeros(T, (dim,))
    mat_temp = zeros(T, (dim, dim))
    vec_temp = zeros(T, (dim,))

    M = typeof(mat)
    V = typeof(vec)
    MT = typeof(mat_temp)
    VT = typeof(vec_temp)
    new{M,V,MT,VT}(mat, vec, mat_temp, vec_temp)
  end
end

function get_mat(lsolver::BackslashLSolver)
  lsolver.mat
end

function get_vec(lsolver::BackslashLSolver)
  lsolver.vec
end

function get_mat_temp(lsolver::BackslashLSolver)
  lsolver.mat_temp
end

function get_vec_temp(lsolver::BackslashLSolver)
  lsolver.vec_temp
end

function lsolve!(lsolver::BackslashLSolver, v)
  copy!(v, lsolver.mat \ lsolver.vec)
  v
end

#############
# LULSolver #
#############
mutable struct LULSolver{M,V,MT,VT,F} <: AbstractLSolver
  const mat::M
  const vec::V
  const mat_temp::MT
  const vec_temp::VT
  fac::F

  function LULSolver(dim::Integer, ::Type{T}) where {T}
    mat = zeros(T, (dim, dim))
    vec = zeros(T, (dim,))
    mat_temp = zeros(T, (dim, dim))
    vec_temp = zeros(T, (dim,))

    for i in 1:dim
      mat[i, i] = 1
    end
    fac = lu(mat)

    M = typeof(mat)
    V = typeof(vec)
    MT = typeof(mat_temp)
    VT = typeof(vec_temp)
    F = typeof(fac)
    new{M,V,MT,VT,F}(mat, vec, mat_temp, vec_temp, fac)
  end
end

function get_mat(lsolver::LULSolver)
  lsolver.mat
end

function get_vec(lsolver::LULSolver)
  lsolver.vec
end

function get_mat_temp(lsolver::LULSolver)
  lsolver.mat_temp
end

function get_vec_temp(lsolver::LULSolver)
  lsolver.vec_temp
end

function lsolve!(lsolver::LULSolver, v)
  lsolver.fac = lu!(lsolver.mat)
  ldiv!(v, lsolver.fac, lsolver.vec)
  v
end
