module SteppersTests

using Test
using OperatorsAndSolvers.AbstractTypes
using OperatorsAndSolvers.LineSearches

T = Float32
ᾱ = T(1)
ϕ(α) = (α - ᾱ)^2 / 2
dϕ(α) = α - ᾱ
op = LineSearchOperator(ϕ, dϕ)

# ConstantStepper
step = T(0.001)
sv = ConstantStepper(step)

us, cache = solve(sv, op, T)
u, = us
@test isapprox(u[1], step)

us, cache = solve!(us, sv, op, cache)
u, = us
@test isapprox(u[1], step)

# ExponentialStepper
step = T(0.001)
γ = T(0.9)
sv = ExponentialStepper(step, γ)

us, cache = solve(sv, op, T)
u, = us
@test isapprox(u[1], step)

step *= γ
us, cache = solve!(us, sv, op, cache)
u, = us
@test isapprox(u[1], step)

end # module SteppersTests
