module ConjugateGradientTests

using Test
using LinearAlgebra
using Random
Random.seed!(123)

using OperatorsAndSolvers.AbstractTypes
using OperatorsAndSolvers.LineSearches
using OperatorsAndSolvers.Systems

n = 3
T = Float32
atol = (T == Float32) ? T(1.0e-5) : T(1.0e-15)

A = rand(T, (n, n))
while isapprox(det(A), 0)
  for i in eachindex(A)
    A[i] = rand(T)
  end
end
# Ensure that A is SPD
A = A' * A
b = rand(T, n)
ū = A \ b

op = LinearSystemOperator(A, b)

maxiter = 100
rtol = atol
config = IterativeSystemSolverConfig(maxiter, rtol, atol)
sv = ConjugateGradientSolver(config)

us, cache = solve(sv, op, T)
u, = us
@test isapprox(u, ū)
residual!(cache.r, op, us)
@test isapprox(norm(cache.r), zero(T); atol)

u = fill!(u, 0)
us = (u,)
cache = reset_cache!(cache, sv, op, us)
us, cache = solve!(us, sv, op, cache)
u, = us
@test isapprox(u, ū)
residual!(cache.r, op, us)
@test isapprox(norm(cache.r), zero(T); atol)

end # module ConjugateGradientTests
