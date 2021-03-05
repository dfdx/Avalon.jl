using MLDatasets
using MLDataUtils
using Plots
using Images


include("vae2.jl")


function reconstruct(m::VAE, x::AbstractVector)
    x = reshape(x, length(x), 1)    
    x_rec = decode(m, m(x))
    return x_rec
end


function show_pic(x)
    a = reshape(x, 28, 28)'
    return plot(Gray.(a))
end


function show_all(m, X, device, n=1)
    subplots = []
    cpu = CPU()
    for i in rand(1:size(X, 2), n)
        x = X[:, i]
        x_ = reconstruct(m, device(x))
        p = show_pic(cpu(x))
        p_ = show_pic(cpu(x_))
        push!(subplots, p, p_)
    end
    plot(subplots..., layout=(n, 2))
end


function main()
    device = best_available_device()
    m = VAE(
        Sequential(
            Linear(784 => 512), 
            x -> tanh.(x), 
            Linear(512 => 256),
            x -> tanh.(x)),      
        Linear(256 => 128),
        Linear(256 => 128),
        Sequential(
            Linear(128 => 256), 
            x -> tanh.(x), 
            Linear(256 => 512), 
            x -> tanh.(x), 
            Linear(512 => 784), 
            x -> logistic.(x)))
    # m = VAE(
    #     Sequential(
    #         Linear(784 => 400),
    #         x -> tanh.(x)),
    #     Linear(400 => 128),
    #     Linear(400 => 128),
    #     Sequential(
    #         Linear(128 => 400),
    #         x -> tanh.(x), 
    #         Linear(400 => 784), 
    #         x -> logistic.(x)))
    m = m |> device

    X, _ = MNIST.traindata()
    X = convert(Matrix{Float64}, reshape(X, 784, 60000))
    @time m = fit!(m, X, device=device, opt=Adam(; lr=1e-5), n_epochs=100)

    # hypothesis 1: one part of the loss is already minified, another contains error
    # hypothesis 2: mean instead of sum
    # hypothesis 3: enc2mu & enc2s2 are too small
    show_all(m, X, device, 5)
end
