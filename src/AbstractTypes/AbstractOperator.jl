####################
# AbstractOperator #
####################
"""
  AbstractOperator{N,T}

Abstract type for operators with `N` arguments and of type `T`.

# Mandatory methods
- `dim_domain`
- `dim_range`
- `residual!`
- `jacobian!`

# Optional methods
- `directional_jacobian!`
- `allocate_zero`
- `allocate_residual`
- `allocate_jacobian`
- `allocate_directional_jacobian`
"""
abstract type AbstractOperator{N,T<:AbstractOperatorType} end

# OperatorType trait
"""
    OperatorType(op::AbstractOperator) -> AbstractOperatorType

Implement the `OperatorType` trait for an `AbstractOperator`.
"""
OperatorType(op::AbstractOperator) = OperatorType(typeof(op))
OperatorType(::Type{<:AbstractOperator{N,T}}) where {N,T} = T()

# Aliases
"""
    AbstractNonlinearOperator{N}

Nonlinear operator with `N` arguments. Alias for
`AbstractOperator{N,NonlinearOperatorType}`.
"""
const AbstractNonlinearOperator{N} =
  AbstractOperator{N,NonlinearOperatorType}

"""
    AbstractQuasilinearOperator{N}

Quasilinear operator with `N` arguments. Alias for
`AbstractOperator{N,QuasilinearOperatorType}`.
"""
const AbstractQuasilinearOperator{N} =
  AbstractOperator{N,QuasilinearOperatorType}

"""
    AbstractSemilinearOperator{N}

Semilinear operator with `N` arguments. Alias for
`AbstractOperator{N,SemilinearOperatorType}`.
"""
const AbstractSemilinearOperator{N} =
  AbstractOperator{N,SemilinearOperatorType}

"""
    AbstractLinearOperator{N}

Linear operator with `N` arguments. Alias for
`AbstractOperator{N,LinearOperatorType}`.
"""
const AbstractLinearOperator{N} =
  AbstractOperator{N,LinearOperatorType}

"""
    dim_domain(op::AbstractOperator, k::Val) -> Integer

Return the domain dimension of `k`-th argument of the operator.
"""
function dim_domain(op::AbstractOperator, k::Val)
  @abstractmethod
end

"""
    dim_range(op::AbstractOperator) -> Integer

Return the range dimension of the operator.
"""
function dim_range(op::AbstractOperator)
  @abstractmethod
end

"""
    allocate_residual(op::AbstractOperator, ::Type{T}) -> AbstractVector

Allocate a zero input for the operator.

Default to dense vectors filled with zeros.
"""
function allocate_zero(op::AbstractOperator{N}, ::Type{T}) where {N,T}
  z = ()
  for k in 1:N
    zk = zeros(T, dim_domain(op, Val(k)))
    z = (z..., zk)
  end
  z
end

"""
    allocate_residual(op::AbstractOperator, ::Type{T}) -> AbstractVector

Allocate the residual vector of the operator.

Default to a dense vector filled with zeros.
"""
function allocate_residual(op::AbstractOperator, ::Type{T}) where {T}
  m = dim_range(op)
  r = zeros(T, m)
  r
end

"""
    residual(
      op::AbstractOperator{N},
      us::NTuple{N,AbstractVector}
    ) -> AbstractVector

Allocate and evaluate the residual vector of the operator at `us`.
"""
function residual(
  op::AbstractOperator,
  us::NTuple{N,AbstractVector}
) where {N}
  T = eltype(first(us))
  r = allocate_residual(op, T)
  residual!(r, op, us)
  r
end

"""
    residual!(
      r::AbstractVector, op::AbstractOperator{N},
      us::NTuple{N,AbstractVector}
    ) -> AbstractVector

Evaluate the residual vector of the operator at `us`.
"""
function residual!(
  r::AbstractVector, op::AbstractOperator{N},
  us::NTuple{N,AbstractVector}
) where {N}
  @abstractmethod
end

"""
    allocate_jacobian(op::AbstractOperator, k::Val, ::Type{T}) -> AbstractMatrix

Allocate the jacobian matrix of the operator with respect to the `k`-th
argument.

Default to a dense matrix filled with zeros.
"""
function allocate_jacobian(op::AbstractOperator, k::Val, ::Type{T}) where {T}
  n, m = dim_domain(op, k), dim_range(op)
  J = zeros(T, (m, n))
  J
end

"""
    jacobian(
      op::AbstractOperator{N}, k::Val,
      us::NTuple{N,AbstractVector}
    ) -> AbstractMatrix

Allocate and evaluate the jacobian matrix of the operator with respect to the
`k`-th argument at `us`.
"""
function jacobian(
  op::AbstractOperator{N}, k::Val,
  us::NTuple{N,AbstractVector}
) where {N}
  T = eltype(first(us))
  J = allocate_jacobian(op, k, T)
  jacobian!(J, op, k, us)
  J
end

"""
    jacobian!(
      J::AbstractMatrix, op::AbstractOperator{N}, k::Val,
      us::NTuple{N,AbstractVector}
    ) -> AbstractMatrix

Evaluate the jacobian matrix of the operator with respect to the `k`-th
argument at `us`.
"""
function jacobian!(
  J::AbstractMatrix, op::AbstractOperator{N}, k::Val,
  us::NTuple{N,AbstractVector}
) where {N}
  @abstractmethod
end

"""
    allocate_directional_jacobian(
      op::AbstractOperator, k::Val, ::Type{T}
    ) -> AbstractVector

Allocate the jacobian vector product of the operator with respect to the `k`-th
argument.

Default to a dense vector filled with zeros.
"""
function allocate_directional_jacobian(
  op::AbstractOperator, k::Val, ::Type{T}
) where {T}
  n = dim_domain(op, k)
  j = zeros(T, n)
  j
end

"""
    directional_jacobian(
      op::AbstractOperator{N}, k::Val,
      us::NTuple{N,AbstractVector}, v::AbstractVector
    ) -> AbstractVector

Allocate and evaluate the jacobian of the operator with respect to the `k`-th
argument at `us` in the direction of `v`.
"""
function directional_jacobian(
  op::AbstractOperator{N}, k::Val,
  us::NTuple{N,AbstractVector}, v::AbstractVector
) where {N}
  T = eltype(first(us))
  j = allocate_directional_jacobian(op, k, T)
  J = allocate_jacobian(op, k, T)
  directional_jacobian!(j, J, op, k, us, v)
  (j, J)
end

"""
    directional_jacobian!(
      j::AbstractVector, J, op::AbstractOperator{N}, k::Val,
      us::NTuple{N,AbstractVector}, v::AbstractVector
    ) -> AbstractVector

Evaluate the jacobian of the operator with respect to the `k`-th argument at
`us` in the direction of `v`.

Default to the standard matrix product `j = jacobian(op, k, us) ⋅ v`. Here `J`
is provided as a cache to compute the full jacobian matrix if needed, but
efficient implementations should not make use of it.
"""
function directional_jacobian!(
  j::AbstractVector, J, op::AbstractOperator{N}, k::Val,
  us::NTuple{N,AbstractVector}, v::AbstractVector
) where {N}
  jacobian!(J, op, k, us)
  mul!(j, J, v)
  (j, J)
end
