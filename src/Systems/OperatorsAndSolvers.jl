##########################
# AbstractSystemOperator #
##########################
"""
    AbstractSystemOperator{N,M,T}

Abstract type for systems of `M` equations with `N` unknowns, of the form
``res(u) = 0``, of type `T`.
"""
abstract type AbstractSystemOperator{N,M,T} <:
              AbstractOperator{1,T} end

dim_domain(op::AbstractSystemOperator{N}, ::Val{1}) where {N} = N

dim_range(op::AbstractSystemOperator{N,M}) where {N,M} = M

################################
# AbstractLinearSystemOperator #
################################
"""
    AbstractLinearSystemOperator

Abstract type for a linear system operator, i.e. ``A ⋅ u - b = 0``. Alias for
`AbstractSystemOperator{LinearOperatorType}`.

# Mandatory methods
- `get_matrix`
- `get_vector`
"""
const AbstractLinearSystemOperator{N,M} =
  AbstractSystemOperator{N,M,LinearOperatorType}

"""
    get_matrix(op::AbstractLinearSystemOperator) -> AbstractMatrix

Return the matrix of the linear system.
"""
function get_matrix(op::AbstractLinearSystemOperator)
  @abstractmethod
end

"""
    get_vector(op::AbstractLinearSystemOperator) -> AbstractVector

Return the vector of the linear system.
"""
function get_vector(op::AbstractLinearSystemOperator)
  @abstractmethod
end

# AbstractOperator interface
function residual!(
  r::AbstractVector, op::AbstractLinearSystemOperator,
  us::NTuple{1,AbstractVector}
)
  u, = us
  A, b = get_matrix(op), get_vector(op)
  mul!(r, A, u)
  axpy!(-1, b, r)
  r
end

function allocate_jacobian(
  op::AbstractLinearSystemOperator, ::Val{1}, ::Type{T}
) where {T}
  A = get_matrix(op)
  similar(A, T)
end

function jacobian!(
  J::AbstractMatrix, op::AbstractLinearSystemOperator, ::Val{1},
  us::NTuple{1,AbstractVector}
)
  A = get_matrix(op)
  copy!(J, A)
  J
end

function directional_jacobian!(
  j::AbstractVector, J, op::AbstractLinearSystemOperator, ::Val{1},
  us::NTuple{1,AbstractVector}, v::AbstractVector
)
  A = get_matrix(op)
  mul!(j, A', v)
  (j, J)
end

########################
# LinearSystemOperator #
########################
"""
    LinearSystemOperator

Generic linear system operator ``A ⋅ u - b = 0`` defined by the matrix `A` and
the vector `b`.
"""
struct LinearSystemOperator{N,M,A,B} <:
       AbstractLinearSystemOperator{N,M}
  A::A
  b::B

  function LinearSystemOperator(A::AbstractMatrix, b::AbstractVector)
    N = size(A, 2)
    M = size(A, 1)

    AA = typeof(A)
    B = typeof(b)
    new{N,M,AA,B}(A, b)
  end
end

# AbstractLinearSystemOperator interface
get_matrix(op::LinearSystemOperator) = op.A

get_vector(op::LinearSystemOperator) = op.b

########################
# AbstractSystemSolver #
########################
"""
    AbstractSystemSolver

Abstract type for a solver for systems of equations.
"""
abstract type AbstractSystemSolver <:
              AbstractSolver end
