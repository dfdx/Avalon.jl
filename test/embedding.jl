@testset "embdding" begin
    s = [1 4; 2 1; 1 6]
    emb = Embedding(6 => 4)
    W = emb.W

    f = (W, s) -> sum(embedding(W, s))
    numeric = ngradient2(f, (W, s), 1)    
    @test grad(f, W, s)[2][1] == numeric

    f = (emb, s) -> sum(emb(s))
    @test grad(f, emb, s)[2][1].W == numeric

    if CUDA.functional()
        gpu = GPU(1)
        emb = gpu(emb)
        s = gpu(s)
        numeric = gpu(numeric)
        @test grad(f, emb, s)[2][1].W == numeric
    end
end
