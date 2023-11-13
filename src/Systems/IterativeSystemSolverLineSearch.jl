
##############################################
# LineSearchOperatorForIterativeSystemSolver #
##############################################
"""
    LineSearchOperatorForIterativeSystemSolver

Line search operator involved in an iterative solver for systems of equations,
i.e. ϕ(α) = residual(u - α v).
"""
struct LineSearchOperatorForIterativeSystemSolver{
  U,V,O,ULS,RLS,JMLS,JVLS,Φ0,DΦ0} <:
       AbstractLineSearchOperator
  u::U
  v::V
  subop::O

  uls::ULS
  rls::RLS
  jls::JMLS
  Jls::JVLS

  ϕ0::Φ0
  dϕ0::DΦ0

  function LineSearchOperatorForIterativeSystemSolver(
    op::AbstractSystemOperator, ::Type{T}
  ) where {T}
    u, = allocate_zero(op, T)
    v, = allocate_zero(op, T)

    uls, = allocate_zero(op, T)
    rls = allocate_residual(op, T)
    jls = allocate_directional_jacobian(op, Val(1), T)
    Jls = allocate_jacobian(op, Val(1), T)

    ϕ0 = Ref(zero(T))
    dϕ0 = Ref(zero(T))

    U = typeof(u)
    V = typeof(v)
    O = typeof(op)
    ULS = typeof(uls)
    RLS = typeof(rls)
    JVLS = typeof(jls)
    JMLS = typeof(Jls)
    Φ0 = typeof(ϕ0)
    DΦ0 = typeof(dϕ0)
    new{U,V,O,ULS,RLS,JVLS,JMLS,Φ0,DΦ0}(
      u, v, op,
      uls, rls, jls, Jls,
      ϕ0, dϕ0
    )
  end
end

"""
    update_lsop!(
      op::LineSearchOperatorForIterativeSystemSolver,
      u::AbstractVector, v::AbstractVector, r::AbstractVector
    ) -> LineSearchOperatorForIterativeSystemSolver

Update the point and direction about which the line search is performed.
"""
function update_lsop!(
  op::LineSearchOperatorForIterativeSystemSolver,
  u::AbstractVector, v::AbstractVector, r::AbstractVector
)
  copy!(op.u, u)
  copy!(op.v, v)
  op.ϕ0[] = fastnorm(r) / 2
  op.dϕ0[] = -dot(v, r)
  op
end

# AbstractLineSearchOperator interface
function get_ϕ(op::LineSearchOperatorForIterativeSystemSolver)
  u, v, subop = op.u, op.v, op.subop
  uls, rls = op.uls, op.rls

  function ϕ(α)
    # | residual(u - α v) |^2 / 2
    copy!(uls, u)
    axpy!(-α, v, uls)
    usls = (uls,)

    residual!(rls, subop, usls)
    normsq(rls) / 2
  end

  ϕ
end

function get_dϕ(op::LineSearchOperatorForIterativeSystemSolver)
  u, v, subop = op.u, op.v, op.subop
  uls, rls, jls, Jls = op.uls, op.rls, op.jls, op.Jls

  function dϕ(α)
    # - <residual(u - α v) | directional_jacobian(u - α v, residual(u - α v))>
    copy!(uls, u)
    axpy!(-α, v, uls)
    usls = (uls,)

    residual!(rls, subop, usls)
    directional_jacobian!(jls, Jls, subop, Val(1), usls, rls)
    -dot(rls, jls)
  end

  dϕ
end

function get_ϕ0(op::LineSearchOperatorForIterativeSystemSolver)
  op.ϕ0[]
end

function get_dϕ0(op::LineSearchOperatorForIterativeSystemSolver)
  op.dϕ0[]
end
