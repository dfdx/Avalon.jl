using Lilith
import Lilith.fit!
using Distributions
# using GradDescent
using MLDataUtils
using MLDatasets
# using StatsBase
using ImageView




# variational autoencoder with Gaussian observed and latent variables
mutable struct VAE
    # encoder / recognizer
    enc_l1::Linear          # encoder: layer 1
    enc_l2::Linear          # encoder: layer 2
    enc_l3::Linear          # encoder: mu
    enc_l4::Linear          # encoder: log(sigma^2)
    # decoder / generator
    dec_l1::Linear          # decoder: layer 1
    dec_l2::Linear          # decoder: layer 2
    dec_l3::Linear          # decoder: layer 3
end


function Base.show(io::IO, m::VAE)
    print(io, "VAE($(size(m.enc_l1.W,2)), $(size(m.enc_l1.W,1)), $(size(m.enc_l2.W,1)), " *
          "$(size(m.enc_l3.W,1)), $(size(m.dec_l2.W,1)), $(size(m.dec_l2.W,1)), $(size(m.dec_l3.W,1)))")
end


VAE(n_inp, n_he1, n_he2, n_z, n_hd1, n_hd2, n_out) =
    VAE(
        # encoder
        Linear(n_inp, n_he1),
        Linear(n_he1, n_he2),
        Linear(n_he2, n_z),
        Linear(n_he2, n_z),
        # decoder
        Linear(n_z, n_hd1),
        Linear(n_hd1, n_hd2),
        Linear(n_hd2, n_out)
    )


function encode(m::VAE, x)
    he1 = softplus.(m.enc_l1(x))
    he2 = softplus.(m.enc_l2(he1))
    mu = m.enc_l3(he2)
    log_sigma2 = m.enc_l4(he2)
    return mu, log_sigma2
end

function decode(m::VAE, z)
    hd1 = softplus.(m.dec_l1(z))
    hd2 = softplus.(m.dec_l2(hd1))
    x_rec = logistic.(m.dec_l3(hd2))
    return x_rec
end


function vae_cost(m::VAE, eps, x)
    mu, log_sigma2 = encode(m, x)
    z = mu .+ sqrt.(exp.(log_sigma2)) .* eps
    x_rec = decode(m, z)
    # loss
    rec_loss = -sum(x .* log.(1e-10 .+ x_rec) .+ (1 .- x) .* log.(1e-10 + 1.0 .- x_rec); dims=1)  # BCE
    KLD = -0.5 .* sum(1 .+ log_sigma2 .- mu .^ 2.0f0 - exp.(log_sigma2); dims=1)
    cost = mean(rec_loss .+ KLD)
end


function fit!(m::VAE, X::AbstractMatrix{T};
              n_epochs=50, batch_size=100, opt=SGD(1e-4; momentum=0)) where T
    for epoch in 1:n_epochs
        print("Epoch $epoch: ")
        epoch_cost = 0
        t = @elapsed for (i, x) in enumerate(eachbatch(X, size=batch_size))
            eps = typeof(x)(rand(Normal(0, 1), size(m.enc_l3.W, 1), batch_size))
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


function show_recon(m, x)
    x_ = reconstruct(m, x)
    show_pic(x)
    show_pic(x_)
end


function main()
    m = VAE(784, 500, 500, 20, 500, 500, 784)

    X, _ = MNIST.traindata()
    X = convert(Matrix{Float64}, reshape(X, 784, 60000))
    @time m = fit!(m, X)

    # check reconstructed image
    for i=1:2:10
        show_recon(m, X[:, i])
    end
end
