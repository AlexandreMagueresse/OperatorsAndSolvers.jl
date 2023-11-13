##############################
# AbstractLineSearchOperator #
##############################
"""
    AbstractLineSearchOperator

Abstract type for line search problems, i.e. if ϕ: [0, c[ -> R is a
differentiable objective function such that ϕ'(0) < 0, we want to find
0 < α ≤ c that minimises ϕ.

# Mandatory methods
- `get_ϕ`
- `get_dϕ`

# Optional methods
- `get_ϕ0`
- `get_dϕ0`
"""
abstract type AbstractLineSearchOperator <:
              AbstractNonlinearOperator{1} end

"""
    get_ϕ(op::AbstractLineSearchOperator) -> Function

Evaluate the objective function.
"""
function get_ϕ(op::AbstractLineSearchOperator)
  @abstractmethod
end

"""
    get_dϕ(op::AbstractLineSearchOperator) -> Function

Evaluate the derivative of the objective function.
"""
function get_dϕ(op::AbstractLineSearchOperator)
  @abstractmethod
end

"""
    get_ϕ0(op::AbstractLineSearchOperator) -> Number

Return the value of the objective function at zero.

Default to `get_ϕ(op)(0)`.
"""
function get_ϕ0(op::AbstractLineSearchOperator)
  get_ϕ(op)(0)
end

"""
    get_dϕ0(op::AbstractLineSearchOperator) -> Number

Return the value of the derivative of the objective function at zero.

Default to `get_dϕ(op)(0)`.
"""
function get_dϕ0(op::AbstractLineSearchOperator)
  get_dϕ(op)(0)
end

# AbstractOperator interface
dim_domain(op::AbstractLineSearchOperator, ::Val{1}) = 1

dim_range(op::AbstractLineSearchOperator) = 1

function residual!(
  r::AbstractVector, op::AbstractLineSearchOperator,
  us::NTuple{1,AbstractVector}
)
  u, = us
  α = u[1]
  r[1] = get_ϕ(op)(α)
  r
end

function jacobian!(
  J::AbstractMatrix, op::AbstractLineSearchOperator, ::Val{1},
  us::NTuple{1,AbstractVector}
)
  u, = us
  α = u[1]
  J[1, 1] = get_dϕ(op)(α)
  J
end

function directional_jacobian!(
  j::AbstractVector, J, op::AbstractLineSearchOperator, ::Val{1},
  us::NTuple{1,AbstractVector}, v::AbstractVector
)
  u, = us
  α, β = u[1], v[1]
  j[1] = get_dϕ(op)(α) * β
  (j, J)
end

######################
# LineSearchOperator #
######################
"""
    LineSearchOperator

Generic operator for the line search problem.
"""
struct LineSearchOperator{Φ,DΦ,Φ0,DΦ0} <:
       AbstractLineSearchOperator
  ϕ::Φ
  dϕ::DΦ
  ϕ0::Φ0
  dϕ0::DΦ0

  function LineSearchOperator(ϕ, dϕ, ϕ0, dϕ0)
    Φ = typeof(ϕ)
    DΦ = typeof(dϕ)
    Φ0 = typeof(ϕ0)
    DΦ0 = typeof(dϕ0)
    new{Φ,DΦ,Φ0,DΦ0}(ϕ, dϕ, ϕ0, dϕ0)
  end
end

function LineSearchOperator(ϕ, dϕ)
  ϕ0 = ϕ(0)
  dϕ0 = dϕ(0)
  LineSearchOperator(ϕ, dϕ, ϕ0, dϕ0)
end

# AbstractLineSearchOperator interface
get_ϕ(op::LineSearchOperator) = op.ϕ

get_dϕ(op::LineSearchOperator) = op.dϕ

get_ϕ0(op::LineSearchOperator) = op.ϕ0

get_dϕ0(op::LineSearchOperator) = op.dϕ0

############################
# AbstractLineSearchSolver #
############################
"""
    AbstractLineSearchSolver

Abstract type for a for line search solver.
"""
abstract type AbstractLineSearchSolver <:
              AbstractSolver end
