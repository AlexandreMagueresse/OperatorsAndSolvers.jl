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

end # module SystemsTests
