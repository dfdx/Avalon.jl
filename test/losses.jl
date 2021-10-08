@testset "losses" begin
    ŷ = collect(reshape(0.1:0.1:1.2, 3, 4))
    c = [1, 1, 3, 2]
    @test nllloss(ŷ, c) == -0.625
    test_rrule(nllloss, ŷ, c)
end
