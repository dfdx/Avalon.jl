@testset "activations" begin
    x = randn(4, 5);

    @test gradcheck(x -> sum(logistic.(x)), x)
    @test gradcheck(x -> sum(softplus.(x)), x)
    @test gradcheck(x -> sum(softsign.(x)), x)
    @test gradcheck(x -> sum(logsigmoid.(x)), x)
    @test gradcheck(x -> sum(relu.(x)), x)
    @test gradcheck(x -> sum(leakyrelu.(x, 0.01)), x)
    @test gradcheck(x -> sum(Lilith.elu.(x, 1)), x)
    
    @test gradcheck(x -> sum(softmax(x)), x)
    @test gradcheck(x -> sum(logsoftmax(x)), x)
end
