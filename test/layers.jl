@testset "layers" begin

    # Linear
    l = Linear(5, 4); x = rand(5, 10)
    gp = grad((W, b, x) -> sum(W * x .+ b), l.W, l.b, x)[2]
    gl = grad((l, x) -> sum(l(x)), l, x)[2]
    @test gp[1] == gl[1].W
    @test gp[2] == gl[1].b

    # Sequential
    l1 = Linear(5, 4); l2 = Linear(4, 3); s = Sequential(l1, l2); x = rand(5, 10)
    gp = grad((l1, l2, x) -> sum(l2(l1(x))), s.seq[1], s.seq[2], x)[2]
    gs = grad((s, x) -> sum(s(x)), s, x)[2]
    @test gp[1].W == gs[1].seq[1].W
    @test gp[2].b == gs[1].seq[2].b

    # Conv1d
    x = rand(7, 3, 10); c = Conv1d(3 => 5, 3)
    @test grad((x, w) -> sum(conv1d(x, w)), x, c.W)[2][2] == grad((x, c) -> sum(c(x)), x, c)[2][2].W
    
    # Conv2d
    x = rand(7, 7, 3, 10); c = Conv2d(3 => 5, 3)
    @test grad((x, w) -> sum(conv2d(x, w)), x, c.W)[2][2] == grad((x, c) -> sum(c(x)), x, c)[2][2].W

    # Conv3d
    x = rand(7, 7, 7, 3, 10); c = Conv3d(3 => 5, 3)
    @test grad((x, w) -> sum(conv3d(x, w)), x, c.W)[2][2] == grad((x, c) -> sum(c(x)), x, c)[2][2].W

    # NLLLoss
    x = rand(5, 4); x = log.(x ./ sum(x; dims=1)); c = [3, 2, 1, 4, 5]
    @test grad((x, c) -> nllloss(x, c), x, c)[2][1] == grad((x, c) -> NLLLoss()(x, c), x, c)[2][1]

    # CrossEntropyLoss
    x = rand(5, 4); c = [3, 2, 1, 4, 5]
    @test (grad((x, c) -> crossentropyloss(x, c), x, c)[2][1] ==
           grad((x, c) -> CrossEntropyLoss()(x, c), x, c)[2][1])

    # MSELoss
    x = rand(5, 4); x_target = rand(5, 4)
    @test (grad((x, x_target) -> mseloss(x, x_target), x, x_target)[2][1] ==
           grad((x, x_target) -> MSELoss()(x, x_target), x, x_target)[2][1])
    
end
