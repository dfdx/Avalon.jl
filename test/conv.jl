# on Travis, testing convolutions takes more than 10 minutes and CI stops the build
# to avoid it, we run a background thread that simply prints a message every 30 seconds
# until testint of convolutions is finished
const STOP_ALIVE_ACTIVE = Ref(false)

function show_alive(; sleep_time=30)
    global STOP_ALIVE_ACTIVE[] = true
    Base.Threads.@spawn while STOP_ALIVE_ACTIVE[]
        println("I'm working...")
        Base.Threads.sleep(sleep_time)
    end
end

function stop_show_alive()
    STOP_ALIVE_ACTIVE[] = false
end


@testset "conv" begin    
    show_alive()
    
    # 1D
    x = rand(7, 3, 10); w = rand(3, 3, 1)
    @test gradcheck((x, w) -> sum(conv1d(x, w)), x, w; tol=1e-4)
    @test gradcheck((x, w) -> sum(conv1d(x, w; padding=1, stride=2)), x, w; tol=1e-4)
    
    # 2D
    x = rand(7, 7, 3, 10); w = rand(3, 3, 3, 1)
    @test gradcheck((x, w) -> sum(conv2d(x, w)), x, w; tol=1e-4)
    @test gradcheck((x, w) -> sum(conv2d(x, w; padding=1, stride=2)), x, w; tol=1e-4)
    
    # 3D
    x = rand(7, 7, 7, 3, 10); w = rand(3, 3, 3, 3, 1)
    @test gradcheck((x, w) -> sum(conv3d(x, w)), x, w; tol=1e-4)
    @test gradcheck((x, w) -> sum(conv3d(x, w; padding=1, stride=2)), x, w; tol=1e-4)

    stop_show_alive()
end

@testset "pooling" begin
    x = rand(7, 7, 3, 10);
    @test gradcheck(x -> sum(maxpool2d(x, 2)), x)
    @test gradcheck(x -> sum(maxpool2d(x, 2; stride=2)), x)
end
