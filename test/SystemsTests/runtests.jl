module SystemsTests

using Test

@testset "DirectSolvers" begin
  include("DirectSolversTests.jl")
end

@testset "NewtonRaphson" begin
  include("NewtonRaphsonTests.jl")
end

@testset "GradientDescent" begin
  include("GradientDescentTests.jl")
end

@testset "ConjugateGradient" begin
  include("ConjugateGradientTests.jl")
end

end # module SystemsTests
