@testset "broadcasted" begin
    @test gradcheck(x -> sum(tanh.(x)), rand(2))
end