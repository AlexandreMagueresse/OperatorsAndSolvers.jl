########################
# AbstractOperatorType #
########################
"""
    AbstractOperatorType

Abstract trait that encodes the type of operator.
"""
abstract type AbstractOperatorType end

"""
    NonlinearOperatorType

Generic trait for a nonlinear operator.
"""
struct NonlinearOperatorType <:
       AbstractOperatorType end

###################################
# AbstractQuasilinearOperatorType #
###################################
"""
    AbstractQuasilinearOperatorType

Abstract trait for a quasilinear operator, i.e. an operator whose residual is
linear in its last argument.

# Example
  res(t, u, u̇) = M(t, u) u̇ + f(t, u)
"""
abstract type AbstractQuasilinearOperatorType <:
              AbstractOperatorType end

"""
    QuasilinearOperatorType

Generic trait for a quasilinear operator.
"""
struct QuasilinearOperatorType <:
       AbstractQuasilinearOperatorType end

##################################
# AbstractSemilinearOperatorType #
##################################
"""
    AbstractSemilinearOperatorType

Abstract trait for a semilinear operator, i.e. an operator whose residual is
linear in its last argument, with linear coefficients independent of other
arguments.

# Example
  res(t, u, u̇) = M(t) u̇ + f(t, u)
"""
abstract type AbstractSemilinearOperatorType <:
              AbstractQuasilinearOperatorType end

"""
    SemilinearOperatorType

Generic trait for a semilinear operator.
"""
struct SemilinearOperatorType <:
       AbstractSemilinearOperatorType end

##############################
# AbstractLinearOperatorType #
##############################
"""
    AbstractLinearOperatorType

Abstract trait for a linear operator, i.e. an operator whose residual is linear
in all its arguments.

# Example
  res(t, u, u̇) = M(t) u̇ + K(t) u + f(t)
"""
abstract type AbstractLinearOperatorType <:
              AbstractSemilinearOperatorType end

"""
    LinearOperatorType

Generic trait for a linear operator.
"""
struct LinearOperatorType <:
       AbstractLinearOperatorType end
