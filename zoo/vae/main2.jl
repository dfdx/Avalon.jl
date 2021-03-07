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


function show_recon(m, X, device; n=5)
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


function show_latent_var(m, z_idx, device)
    vals = collect(-1:0.1:1)
    z_len = length(m.enc2mu.b)  # assuming Linear
    z = zeros(z_len, length(vals))
    for i in 1:size(z, 2)
        z[z_idx, i] = vals[i]
    end
    z = device(z)
    x_ = decode(m, z) |> CPU()
    subplots = [show_pic(x_[:, i]) for i in 1:size(x_, 2)]
    plot(subplots..., layout=length(subplots))
end


function show_samples(m, n, device)    
    z_len = length(m.enc2mu.b)  # assuming Linear
    z = randn(z_len, n)
    z = device(z)
    x_ = decode(m, z) |> CPU()
    subplots = [show_pic(x_[:, i]) for i in 1:size(x_, 2)]
    plot(subplots..., layout=length(subplots))
end


function main()
    device = best_available_device()    
    m = VAE(
        Sequential(
            Linear(784 => 400), 
            x -> relu.(x)),
        Linear(400 => 20),
        Linear(400 => 20),
        Sequential(
            Linear(20 => 400), 
            x -> relu.(x),             
            Linear(400 => 784), 
            x -> logistic.(x));
        beta=10)
    m = m |> device

    X, _ = MNIST.traindata()
    X = convert(Matrix{Float64}, reshape(X, 784, 60000))
    @time m = fit!(m, X, device=device, opt=Adam(; lr=1e-3), n_epochs=10)

    show_recon(m, X, device, n=5)
    show_latent_var(m, 1, device)
    show_samples(m, 10, device)
end