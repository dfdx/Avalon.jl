using MLDatasets
using MLDataUtils
using Plots
using Images
using Interact


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


function interpolate_latent_var(m, x, z_idx, device)
    vals = collect(-2:0.5:2)
    z = m(x)
    xs_ = []
    cpu = CPU()
    for v in vals
        z[z_idx, :] = v
        x_ = decode(m, device(z)) |> cpu
        push!(xs_, x_)
    end
    subplots = [show_pic(x_) for x_ in xs_]
    # plot(subplots..., layout=length(subplots))
    return subplots
end


function show_latent_vars(m, x, z_idxs, device)
    groups = [interpolate_latent_var(m, x, z_idx, device) for z_idx in z_idxs]
    subplots = vcat(groups...)
    n_cols = length(groups[1])
    plot(subplots..., layout=(length(z_idxs), n_cols))
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
        beta=5)
    m = m |> device

    X, _ = MNIST.traindata()
    X = convert(Matrix{Float64}, reshape(X, 784, 60000))
    @time m = fit!(m, X, device=device, opt=Adam(; lr=1e-3), n_epochs=10)

    show_recon(m, X, device, n=5)
    show_latent_vars(m, device(X[:, 2]), 1:4, device)
    show_samples(m, 10, device)
end