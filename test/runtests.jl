module OperatorsAndSolversTests

using Test

@time @testset "Helpers" begin
  include("HelpersTests/runtests.jl")
end

@time @testset "AbstractTypes" begin
  include("AbstractTypesTests/runtests.jl")
end

@time @testset "LineSearches" begin
  include("LineSearchesTests/runtests.jl")
end

@time @testset "Systems" begin
  include("SystemsTests/runtests.jl")
end

end # module OperatorsAndSolversTests
