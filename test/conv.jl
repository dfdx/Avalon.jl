import Avalon: convNd

@testset "conv" begin
    kwargs = (padding=1, stride=2)
    for (x_sz, w_sz) in [
            ((8, 3, 10), (3, 3, 1)),              # 1D
            ((8, 8, 3, 10), (3, 3, 3, 1)),        # 2D
        ]
        x = rand(Float32, x_sz); w = rand(Float32, w_sz)
        test_rrule(convNd, x, w; rtol=1e-3, atol=1e-3, check_inferred=false)
        test_rrule(convNd, x, w; fkwargs=kwargs, rtol=1e-3, atol=1e-3, check_inferred=false)
    end

    # 3D is very noisy for numeric gradient, using larger atol & rtol for it
    x = rand(Float32, 8, 8, 8, 3, 1); w = rand(Float32, 3, 3, 3, 3, 1)
    test_rrule(convNd, x, w; rtol=1e-1, atol=1e-1, check_inferred=false)
    test_rrule(convNd, x, w; fkwargs=kwargs, rtol=1e-1, atol=1e-1, check_inferred=false)
end

@testset "pooling" begin
    x = rand(7, 7, 3, 10);
    test_rrule(maxpool2d, x, 2; check_inferred=false)
    test_rrule(maxpool2d, x, 2; fkwargs=(stride=2,), check_inferred=false)
end
