using Optimisation
using Optim
using LineSearches

using LinearAlgebra

using Plots
using Random
using BenchmarkTools
using Printf

Random.seed!(123)
T = Float64

# test = "linear_system"
test = "camel"
# test = "himmelblau"
# test = "matyas"
# test = "rastrigin"
# test = "rosenbrock"
# test = "styblinski_tang"

do_benchmark = false

if test == "linear_system"
  n = 10

  A = 2 * rand(T, n, n) .- 1
  A = A' * A
  b = 2 * rand(T, n) .- 1
  cache = similar(b)

  function objective(x::AbstractVector)
    T = eltype(x)
    res = dot(x, A, x) / 2 - dot(x, b)
    res::T
  end

  function gradient(x::AbstractVector)
    T = typeof(x)
    mul!(cache, A, x)
    cache .-= b
    cache::T
  end

  xtheo = [A \ b]
elseif test == "camel"
  n = 2

  function objective(xs::AbstractVector)
    x, y = xs[1], xs[2]
    T = eltype(xs)

    2 * x^2 - T(1.05) * x^4 + x^6 / 6 + x * y + y^2
  end

  function gradient(xs::AbstractVector)
    x, y = xs[1], xs[2]
    T = eltype(xs)

    [
      4 * x - T(4.2) * x^3 + x^5 + y,
      x + 2 * y
    ]
  end

  xtheo = [zeros(n)]
elseif test == "himmelblau"
  n = 2

  function objective(xs::AbstractVector)
    x, y = xs[1], xs[2]

    (x^2 + y - 11)^2 + (x + y^2 - 7)^2
  end

  function gradient(xs::AbstractVector)
    x, y = xs[1], xs[2]

    [
      4 * x * (x^2 + y - 11) + 2 * (x + y^2 - 7),
      2 * (x^2 + y - 11) + 4 * y * (x + y^2 - 7)
    ]
  end

  xtheo = [
    T[3, 2],
    T[-2.805118, 3.131312],
    T[-3.779310, -3.283186],
    T[3.584428, -1.848126],
  ]
elseif test == "matyas"
  n = 2

  function objective(xs::AbstractVector)
    x, y = xs[1], xs[2]
    T = eltype(xs)

    T(0.26) * (x^2 + y^2) - T(0.48) * x * y
  end

  function gradient(xs::AbstractVector)
    x, y = xs[1], xs[2]
    T = eltype(xs)

    [
      T(0.52) * x - T(0.48) * y,
      T(0.52) * y - T(0.48) * x
    ]
  end

  xtheo = [
    T[0, 0],
  ]
elseif test == "rastrigin"
  n = 10
  A = 10

  # function objective(xs::AbstractVector)
  #   T = eltype(x)
  #   s = zero(T) + A * length(xs)
  #   for x in xs
  #     s +=
  #   end
  #   s
  # end

  # function gradient(x::AbstractVector)
  #   g = zero(x)
  #   g
  # end

  xtheo = [zeros(T, n)]
elseif test == "rosenbrock"
  n = 2

  function objective(x::AbstractVector)
    T = eltype(x)
    s = zero(T)
    n = length(x)
    for i in 1:n-1
      s += 100 * (x[i+1] - x[i]^2)^2 + (1 - x[i])^2
    end
    s
  end

  function gradient(x::AbstractVector)
    g = zero(x)
    n = length(x)
    for i in 1:n-1
      g[i] += -400 * x[i] * (x[i+1] - x[i]^2) - 2 * (1 - x[i])
      g[i+1] += 200 * (x[i+1] - x[i]^2)
    end
    g
  end

  xtheo = [ones(T, n)]
elseif test == "styblinski_tang"
  n = 2
  cache = zeros(n)

  function objective(x::AbstractVector)
    s = zero(eltype(x))
    for xi in x
      s += xi^4 - 16 * xi^2 + 5 * xi
    end
    s / 2
  end

  function gradient(x::AbstractVector)
    for i in eachindex(x)
      xi = x[i]
      cache[i] = 4 * xi^3 - 32 * xi + 5
    end
    cache / 2
  end

  xtheo = [ones(T, n) * T(-2.903534)]
else
  throw("unknown test")
end

optimiser = Gradient()
ls = Backtracking(
  StrongValueCondition(T(1.0e-4)), T,
  # ρ_max=T(0.5),
)

# Initialisation
x = 2 * rand(T, n) .- 1
x0 = copy(x)
temp = similar(x)

# Loop
epochs = 100

show_trace = false

function make_fun_der(x, dir, temp)
  function fun(s)
    temp .= x .+ s * dir
    objective(temp)
  end
  function der(s)
    temp .= x .+ s * dir
    gradient(temp)
  end
  fun, der
end

function loop(objective, gradient, x0, temp, epochs, optimiser, show_trace=false)
  histval = zeros(T, epochs + 1)
  x = copy(x0)

  for epoch in 1:epochs
    # Obtain direction
    grad = gradient(x)
    dir = direction(optimiser, grad)

    # Test convergence
    if isapprox(norm(grad), 0)
      break
    end

    # Perform line search
    fun, der = make_fun_der(x, dir, temp)

    val = objective(x)
    slope = dot(grad, dir)

    stepinit = one(T)
    ss = SearchState(fun, der, val, slope, stepinit, T(Inf))
    state = search(ls, ss)
    step = get_step(state)

    # Test convergence
    if isapprox(step * norm(grad), 0)
      break
    end

    # Update
    x .+= step * dir

    # Report
    histval[epoch] = val
    if show_trace
      println(@sprintf "%5i %.3e %.3e" epoch val norm(grad))
    end
  end
  histval[epochs+1] = objective(x)

  x, histval
end

################
# Optimisation #
################
xbest, histval = loop(objective, gradient, x, temp, epochs, optimiser, show_trace)
valbest = histval[end]

valdist = abs(valbest - objective(xtheo[1]))
xdist = minimum(norm(xbest - xtheoi) for xtheoi in xtheo)

println("Custom")
println("------")
println(@sprintf "Δv\t%.5e" valdist)
println(@sprintf "Δx\t%.5e" xdist)

#########
# Optim #
#########
gradient!(G, x) = copy!(G, gradient(x))
method = GradientDescent(
  alphaguess=InitialStatic(),
  linesearch=BackTracking(c_1=ls.cond_value.c)
)
options = Optim.Options(
  iterations=epochs - 1,
  show_trace=show_trace
)

# Clear time
res = optimize(
  objective, gradient!, x0,
  method, options
)
valbest = res.minimum
xbest = res.minimizer

valdist = abs(valbest - objective(xtheo[1]))
xdist = minimum(norm(xbest - xtheoi) for xtheoi in xtheo)

println()
println("Optim")
println("-----")
println(@sprintf "Δv\t%.5e" valdist)
println(@sprintf "Δx\t%.5e" xdist)

#############
# Benchmark #
#############
if do_benchmark
  println()
  println("Benchmark")
  println("---------")

  stats = @benchmark loop(
    $objective, $gradient, $x0,
    $temp, $epochs, $optimiser, false)
  println(@sprintf "Custom\t%.3e" mean(stats.times))

  options = Optim.Options(
    iterations=epochs - 1,
    show_trace=false
  )
  stats = @benchmark optimize(
    $objective, $gradient!, $x0,
    $method, $options
  )
  println(@sprintf "Optim\t%.3e" mean(stats.times))
end
