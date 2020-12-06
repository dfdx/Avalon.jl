device = best_available_device()
cpu = CPU()


@testset "cuda: conv" begin
    # 1D
    x = rand(7, 3, 10); w = rand(3, 3, 1)
    g = grad((x, w) -> sum(conv1d(x, w)), x, w)[2]
    d_g = grad((x, w) -> sum(conv1d(x, w)), device(x), device(w))[2]
    @test g[1] ≈ cpu(d_g[1])
    @test g[2] ≈ cpu(d_g[2])

    
    # 2D
    x = rand(7, 7, 3, 10); w = rand(3, 3, 3, 1)
    g = grad((x, w) -> sum(conv2d(x, w)), x, w)[2]
    d_g = grad((x, w) -> sum(conv2d(x, w)), device(x), device(w))[2]
    @test g[1] ≈ cpu(d_g[1])
    @test g[2] ≈ cpu(d_g[2])

    # 3D
    x = rand(7, 7, 7, 3, 10); w = rand(3, 3, 3, 3, 1)
    g = grad((x, w) -> sum(conv3d(x, w)), x, w)[2]
    d_g = grad((x, w) -> sum(conv3d(x, w)), device(x), device(w))[2]
    @test g[1] ≈ cpu(d_g[1])
    @test g[2] ≈ cpu(d_g[2])
end

@testset "cuda: pooling" begin
    x = rand(7, 7, 3, 10);
    g = grad(x -> sum(maxpool2d(x, 2)), x)[2]
    d_g = grad(x -> sum(maxpool2d(x, 2)), device(x))[2]
    @test g[1] ≈ cpu(d_g[1])
end


@testset "cuda: batchnorm" begin
    Random.seed!(898);

    m = BatchNorm2d(3)
    x = rand(10, 10, 3, 5)
    d_m = device(m)
    d_x = device(x)

    y = m(x)
    d_y = d_m(d_x)
    @test isapprox(cpu(d_y), y; rtol=1e-2)

    dy = randn(size(x)...)
    d_dy = device(dy)

    dgamma, dbeta, dx = ∇batchnorm2d(dy, m, m.gamma, m.beta, x)
    d_dgamma, d_dbeta, d_dx = ∇batchnorm2d(d_dy, d_m, d_m.gamma, d_m.beta, d_x)
    @test isapprox(cpu(d_dgamma), dgamma; rtol=1e-2)
    @test isapprox(cpu(d_dbeta), dbeta; rtol=1e-2)
    @test isapprox(cpu(d_dx), dx; rtol=1e-2)

    _, g = grad((m, x) -> sum(m(x)), m, x)
    _, d_g = grad((m, x) -> sum(m(x)), d_m, d_x)

    @test isapprox(g[1].gamma, cpu(d_g[1].gamma); rtol=1e-2, atol=1e-4)
    @test isapprox(g[1].beta, cpu(d_g[1].beta); rtol=1e-2, atol=1e-4)
    @test isapprox(g[2], cpu(d_g[2]); rtol=1e-2, atol=1e-4)

end


@testset "cuda: RNN" begin
    Random.seed!(128);

    # vanilla RNN
    m = RNN(10 => 5); x_seq = ones(10, 4, 10); h = init_hidden(m, 4)
    _, g = grad((m, x_seq, h) -> begin h_all, h = m(x_seq, h); sum(h_all) end, m, x_seq, h)
    d_m, d_x_seq, d_h = map(device, (m, x_seq, h))
    _, d_g = grad((m, x_seq, h) -> begin h_all, h = m(x_seq, h); sum(h_all) end, d_m, d_x_seq, d_h)
    @test g[1].cell.W_ih ≈ cpu(d_g[1].cell.W_ih)

    # LSTM
    m = LSTM(10 => 5); x_seq = ones(10, 4, 10); h, c = init_hidden(m, 4)
    _, g = grad((m, x_seq, h, c) -> begin h_all, h, c = m(x_seq, h, c); sum(h) end, m, x_seq, h, c)
    d_m, d_x_seq, d_h, d_c = map(device, (m, x_seq, h, c))
    _, d_g = grad((m, x_seq, h, c) -> begin h_all, h, c = m(x_seq, h, c); sum(h) end, d_m, d_x_seq, d_h, d_c)
    @test g[1].cell.W_ih ≈ cpu(d_g[1].cell.W_ih)

    # GRU
    m = GRU(10 => 5); x_seq = ones(10, 4, 10); h = init_hidden(m, 4)
    _, g = grad((m, x_seq, h) -> begin h_all, h = m(x_seq, h); sum(h_all) end, m, x_seq, h)
    d_m, d_x_seq, d_h = map(device, (m, x_seq, h))
    _, d_g = grad((m, x_seq, h) -> begin h_all, h = m(x_seq, h); sum(h_all) end, d_m, d_x_seq, d_h)
    @test g[1].cell.W_ih ≈ cpu(d_g[1].cell.W_ih)

end


@testset "cuda: activations" begin
    x = rand(Float32, 5, 5);    
    d_x = device(x);

    @test grad(x -> sum(logistic.(x)), x)[2][1] ≈ grad(x -> sum(logistic.(x)), d_x)[2][1] |> cpu
    @test grad(x -> sum(softplus.(x)), x)[2][1] ≈ grad(x -> sum(softplus.(x)), d_x)[2][1] |> cpu
    @test grad(x -> sum(softsign.(x)), x)[2][1] ≈ grad(x -> sum(softsign.(x)), d_x)[2][1] |> cpu
    @test grad(x -> sum(relu.(x)), x)[2][1] ≈ grad(x -> sum(relu.(x)), d_x)[2][1] |> cpu
    @test grad(x -> sum(leakyrelu.(x, 0.01)), x)[2][1] ≈
        grad(x -> sum(leakyrelu.(x, 0.01)), d_x)[2][1] |> cpu
    # ELU on CUDA results in scalar operations warning followed by segfault, disabling it for now
    # @test grad(x -> sum(elu.(x, 1.0)), x)[2][1] ≈ grad(x -> sum(elu.(x, 1.0)), d_x)[2][1]

    g = grad(x -> sum(softmax(x)), x)[2][1]
    d_g = grad(x -> sum(softmax(x)), d_x)[2][1]
    @test isapprox(g, cpu(d_g), rtol = 1e-5, atol = 1e-5)

    x = rand(Float32, 5, 5);    
    d_x = device(x);
    @test grad(x -> sum(logsoftmax(x)), x)[2][1] ≈ grad(x -> sum(logsoftmax(x)), d_x)[2][1] |> cpu
end


@testset "cuda: losses" begin
    x = rand(5, 4); x = log.(x ./ sum(x; dims=1)); c = [3, 2, 1, 4, 5]
    d_x = device(x); d_c = device(c)
    @test grad((x, c) -> nllloss(x, c), x, c)[2][1] ≈
        grad((x, c) -> nllloss(x, c), d_x, d_c)[2][1] |> cpu

    x = rand(5, 4); c = [3, 2, 1, 4, 5]
    d_x = device(x); d_c = device(c)
    @test (grad((x, c) -> crossentropyloss(x, c), x, c)[2][1] ≈
           grad((x, c) -> crossentropyloss(x, c), d_x, d_c)[2][1] |> cpu)

    x = rand(5, 4); x_target = rand(5, 4)
    d_x = device(x); d_x_target = device(x_target)
    @test (grad((x, x_target) -> mseloss(x, x_target), x, x_target)[2][1] ≈
           grad((x, x_target) -> mseloss(x, x_target), d_x, d_x_target)[2][1] |> cpu)
end


@testset "cuda: optim" begin

    # not every parameter update will lead to descreased loss
    # but at least we can check that parameters are actually changed
    m = MyModel(Linear(5, 4)) |> device; x = rand(5, 10) |> device;
    old_m = deepcopy(m); old_x = deepcopy(x)
    _, g = grad(my_model_loss, m, x)

    # SGD
    update!(SGD(0.1; momentum=0.5), m, g[1])
    @test old_m.linear.W != m.linear.W
    update!(SGD(0.1; momentum=0.5), x, g[2])
    @test old_x != x

    # RMSprop
    update!(RMSprop(), m, g[1])
    @test old_m.linear.W != m.linear.W
    update!(RMSprop(), x, g[2])
    @test old_x != x

    # Adam
    update!(Adam(), m, g[1])
    @test old_m.linear.W != m.linear.W
    update!(Adam(), x, g[2])
    @test old_x != x

end
