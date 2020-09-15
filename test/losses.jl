@testset "losses" begin
    # NLL - checked only for one-hot encoded matrix since gradcheck doesn't support Vector{Int}
    classes = rand(1:10, 5)
    y = zeros(10, 5)
    for (j, c) in enumerate(classes)
        y[c, j] = 1.0
    end
    ŷ = logsoftmax(rand(10, 5))
    @test gradcheck(nllloss, ŷ, y)
end
