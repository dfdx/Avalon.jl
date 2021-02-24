using Avalon
import Avalon.fit!
import Yota.@nodiff
using Distributions
using MLDatasets
using MLDataUtils
using BenchmarkTools
using ImageView


# function binarycrossentropy(ŷ, y; agg = mean, ϵ = epseltype(ŷ))
#     agg(@.(-xlogy(y, ŷ + ϵ) - xlogy(1 - y, 1 - ŷ + ϵ)))
# end

@nodiff eps(x) x


function binarycrossentropy(ŷ, y; agg=mean, ϵ=eps(eltype(ŷ)))
    agg(@.(-y * log(ŷ + ϵ) - (1 - y) * log(1 - ŷ + ϵ)))
end

function crossentropy(ŷ, y; dims = 1, agg = mean, ϵ = eps(eltype(ŷ)))
    agg(.-sum(y .* log.(ŷ .+ ϵ); dims = dims))
end

function kldivergence(ŷ, y; dims = 1, agg = mean, ϵ = eps(eltype(ŷ)))
    entropy = agg(sum(x .* log.(y), dims = dims))
    cross_entropy = crossentropy(ŷ, y; dims = dims, agg = agg, ϵ = ϵ)
    return entropy + cross_entropy
end


mutable struct VAE
    encoder::Sequential
    enc2mu::Linear
    enc2s::Linear
    decoder::Sequential
end


VAE(encoder::Sequential, z_len::Int, decoder::Sequential) =
    VAE(encoder,
        Linear(z_len => z_len),
        Linear(z_len => z_len),
        decoder)


function Base.show(io::IO, m::VAE)
    print(io, "VAE()")
end


function (m::VAE)(x)
    h = m.encoder(x)
    mu = m.enc2mu(h)
    return mu
end


function encode(m::VAE, x)
    h = m.encoder(x)
    mu = m.enc2mu(h)
    log_s2 = m.enc2s(h)
    return mu, log_s2
end

decode(m::VAE, z) = m.decoder(z)

function loss_function(m::VAE, eps, x)
    mu, log_s2 = encode(m, x)
    z = mu .+ sqrt.(exp.(log_s2)) .* eps
    x_rec = decode(m, z)
    # BCE = -mean(x .* log.(1f-10 .+ x_rec) .+ (1 .- x) .* log.(1f-10 + 1 .- x_rec); dims=1)
    BCE = binarycrossentropy(x_rec, x)
    # TODO: use this instead: https://github.com/FluxML/Flux.jl/blob/d341500501c2d70f69a6a8343cc3a46d7bf43795/src/losses/functions.jl#L364
    # KLD = -0.5f0 .* mean(1 .+ log_s2 .- mu .^ 2f0 - exp.(log_s2); dims=1)
    KLD = kldivergence(x_rec, x)
    return mean(BCE .+ KLD)
end


function fit!(m::VAE, X::AbstractMatrix{T};
              n_epochs=50, batch_size=100, opt=Adam(; lr=1e-4), device=CPU()) where T
    for epoch in 1:n_epochs
        print("Epoch $epoch: ")
        epoch_cost = 0
        t = @elapsed for (i, x) in enumerate(eachbatch(X, size=batch_size))
            x = device(x)
            eps = typeof(x)(rand(Normal(0, 1), size(m.enc2mu.W, 1), batch_size))
            cost, g = grad(loss_function, m, eps, x)
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

# TODO: remove dim argument, check convergence

function main()
    device = best_available_device()
    m = VAE(
        Sequential(Linear(784 => 500), x -> tanh.(x), Linear(500 => 5), x -> tanh.(x)),
        5,
        Sequential(Linear(5 => 500), x -> tanh.(x), Linear(500 => 784), x -> logistic.(x)))
    m = m |> device

    X, _ = MNIST.traindata()
    X = convert(Matrix{Float64}, reshape(X, 784, 60000))
    @time m = fit!(m, X, device=device, opt=Adam(; lr=1e-5), n_epochs=5)

    # check reconstructed image
    for i=1:2:10
        show_recon(m, X[:, i], device)
    end
end
