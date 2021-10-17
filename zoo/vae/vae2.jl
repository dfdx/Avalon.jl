using Avalon
import Avalon.fit!
# import Yota.@nodiff
using Distributions
using MLDataUtils


# @nodiff eps(x) x


function binarycrossentropy(ŷ, y; agg=mean, ϵ=eps(eltype(ŷ)))
    agg(@.(-y * log(ŷ + ϵ) - (1 - y) * log(1 - ŷ + ϵ)))
end

function kldiv_normal(mu, log_s2; agg=mean)
    -1/2 * agg(@. 1 + log_s2 - mu ^ 2f0 - exp(log_s2))
end


mutable struct VAE
    encoder::Sequential
    enc2mu::Linear
    enc2s::Linear
    decoder::Sequential
    beta::Real
end

Avalon.ignored_field(m::VAE, ::Val{(:beta,)}) = true


VAE(encoder::Sequential, enc2mu::Any, enc2s::Any, decoder::Sequential; beta=1) =
    VAE(encoder,
        enc2mu,
        enc2s,
        decoder,
        beta)


VAE(encoder::Sequential, z_len::Int, decoder::Sequential; beta=1) =
    VAE(encoder,
        Linear(z_len => z_len),
        Linear(z_len => z_len),
        decoder,
        beta)


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


function loss_function(m::VAE, ϵ, x)
    mu, log_s2 = encode(m, x)
    z = mu .+ sqrt.(exp.(log_s2)) .* ϵ
    x_rec = decode(m, z)
    BCE = binarycrossentropy(x_rec, x; agg=sum)
    KLD = kldiv_normal(mu, log_s2; agg=sum)
    return BCE + m.beta * KLD
end


function fit!(m::VAE, X::AbstractMatrix{T};
              n_epochs=50, batch_size=100, opt=Adam(; lr=1e-5), device=CPU()) where T
    println("Starting training")
    for epoch in 1:n_epochs
        epoch_cost = 0
        t = @elapsed for (i, x) in enumerate(eachbatch(X, size=batch_size))
            x = device(x)
            ϵ = typeof(x)(rand(Normal(0, 1), size(m.enc2mu.W, 1), batch_size))
            cost, g = grad(loss_function, m, ϵ, x)
            update!(opt, m, g[2])
            epoch_cost += cost
        end
        println("Epoch $epoch: avg_cost=$(epoch_cost / (size(X,2) / batch_size)), elapsed=$t")
    end
    return m
end