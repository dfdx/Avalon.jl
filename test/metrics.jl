@testset "metrics" begin
    y = [1, 1, 0, 0]
    ŷ = [1, 1, 1, 1]
    @test accuracy(y, ŷ) == 0.5
    @test recall(y, ŷ) == 1
    @test precision(y, ŷ) == 0.5

    y = [:d, :c, :a, :b]
    ŷ = [:d, :d, :a, :b]
    @test confusion_matrix(y, ŷ) == [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 0 1]
    
    
end
