using Avalon
import Avalon.fit!
using Distributions
using MLDatasets
using MLDataUtils
using BenchmarkTools
using ImageView


mutable struct VAE
    # encoder / recognizer
    enc_i2h::Linear
    enc_h2mu::Linear
    enc_h2logsigma2::Linear
    # decoder / generator
    dec_z2h::Linear
    dec_h2o::Linear
end


function Base.show(io::IO, m::VAE)
    print(io, "VAE()")
end


VAE(n_i, n_h, n_z) =
    VAE(
        # encoder
        Linear(n_i, n_h),
        Linear(n_h, n_z),
        Linear(n_h, n_z),
        # decoder
        Linear(n_z, n_h),
        Linear(n_h, n_i),
    )


function encode(m::VAE, x)
    he1 = tanh.(m.enc_i2h(x))
    mu = m.enc_h2mu(he1)
    log_sigma2 = m.enc_h2logsigma2(he1)
    return mu, log_sigma2
end

function decode(m::VAE, z)
    hd1 = tanh.(m.dec_z2h(z))
    x_rec = logistic.(m.dec_h2o(hd1))
    return x_rec
end


function vae_cost(m::VAE, eps, x)
    mu, log_sigma2 = encode(m, x)
    z = mu .+ sqrt.(exp.(log_sigma2)) .* eps
    x_rec = decode(m, z)
    # loss
    rec_loss = -sum(x .* log.(1f-10 .+ x_rec) .+ (1 .- x) .* log.(1f-10 + 1 .- x_rec); dims=1)
    KLD = -0.5f0 .* sum(1 .+ log_sigma2 .- mu .^ 2.0f0 - exp.(log_sigma2); dims=1)
    cost = mean(rec_loss .+ KLD)
end


function fit!(m::VAE, X::AbstractMatrix{T};
              n_epochs=50, batch_size=100, opt=SGD(1e-4; momentum=0), device=CPU()) where T
    for epoch in 1:n_epochs
        print("Epoch $epoch: ")
        epoch_cost = 0
        t = @elapsed for (i, x) in enumerate(eachbatch(X, size=batch_size))
            x = device(x)
            eps = typeof(x)(rand(Normal(0, 1), size(m.enc_h2mu.W, 1), batch_size))
            cost, g = grad(vae_cost, m, eps, x)
            update!(opt, m, g[1])
            epoch_cost += cost
        end
        println("avg_cost=$(epoch_cost / (size(X,2) / batch_size)), elapsed=$t")
    end
    return m
end


function reconstruct(m::VAE, x::AbstractVector)
    x = reshape(x, length(x), 1)
    mu, _ = encode(m, x)
    z = mu
    x_rec = decode(m, z)
    return x_rec
end


function show_pic(x)
    reshape(x, 28, 28)' |> imshow
end


function show_recon(m, x, device)
    x_ = reconstruct(m, device(x))
    show_pic(x)
    show_pic(x_)
end


function main()
    device = best_available_device()
    m = VAE(784, 500, 5) |> device

    X, _ = MNIST.traindata()
    X = convert(Matrix{Float64}, reshape(X, 784, 60000))
    @time m = fit!(m, X, device=device)

    # check reconstructed image
    for i=1:2:10
        show_recon(m, X[:, i], device)
    end
end
