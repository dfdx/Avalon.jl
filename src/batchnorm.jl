mutable struct BatchNorm2d
    gamma::AbstractArray
    beta::AbstractArray
    eps::Real
    momentum::Real
    training::Bool
    running_mean::AbstractArray
    running_var::AbstractArray
    count::Int
end

BatchNorm2d(num_features::Int; eps=1e-5, momentum=0.1, training=true) =
    BatchNorm2d(
        ones(1, 1, num_features, 1),
        zeros(1, 1, num_features, 1),
        eps,
        momentum,
        training,
        zeros(1, 1, num_features, 1),
        zeros(1, 1, num_features, 1),
        0
    )

Base.show(io::IO, m::BatchNorm2d) = print(io, "BatchNorm2d($(length(m.beta)))")

trainmode!(m::BatchNorm2d, train::Bool=true) = (m.training = train; nothing)


function batchnorm_impl(gamma::AbstractArray, beta::AbstractArray, x::AbstractArray,
                        mu::AbstractArray, sigma2::AbstractArray, momentum::Real; eps, training)
    x_hat = (x .- mu) ./ sqrt.(sigma2 .+ eps)
    return gamma .* x_hat .+ beta
end


function ∇batchnorm_impl(gamma::AbstractArray, beta::AbstractArray, x::AbstractArray, dy::AbstractArray,
                         mu::AbstractArray, sigma2::AbstractArray, momentum; eps, training)
    # based on:
    # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    sum_dims = tuple([i for i=1:ndims(x) if size(x, i) != size(mu, i)]...)   # all dims but channel
    P = prod(size(x, i) for i in sum_dims)   # product of all dims but channel
    typ = eltype(gamma)
    _1 = one(typ)        # 1 of eltype(x)
    _05 = _1 / 2         # 0.5 of eltype(x)
    # part of forward pass - we can optimize it later and take from cache
    xmu = x .- mu
    sqrtvar = sqrt.(sigma2 .+ eps)
    ivar = one(typ) ./ sqrtvar
    xhat = xmu .* ivar
    # reverse pass
    dbeta = Yota.unbroadcast(beta, dy)
    dgamma = Yota.unbroadcast_prod_x(gamma, xhat, dy)
    dxhat = dy .* gamma
    divar = sum(dxhat .* xmu; dims=sum_dims)
    dxmu1 = dxhat .* ivar
    dsqrtvar = -(sqrtvar .^ -2) .* divar
    dvar = _05 ./ sqrtvar .* dsqrtvar
    dsq = one(typ) ./ P .* dvar
    dxmu2 = 2 .* xmu .* dsq
    dx1 = (dxmu1 .+ dxmu2)
    dmu = -sum(dxmu1 .+ dxmu2; dims=sum_dims)
    dx2 = Yota.∇mean(x, dmu, sum_dims)
    dx = dx1 .+ dx2
    return dgamma, dbeta, dx
end


batch_mean(x, dims) = mean(x, dims=dims)
function batch_var(x, dims)
    # note that this is equivalent to Statistics.var(x; dims=dims, corrected=false)
    # but also works for CuArrays
    mu = mean(x, dims=dims)
    return mean((x .- mu) .^ 2; dims=dims)
end


# BatchNorm layer is somewhat tricky: on one hand, we want to track running mean and var during
# training, and since these are mutating operations, we need to hide them inside primitives -
# something like `forward(m:BatchNorm, x)`. On other hand, Yota doesn't support diffrules
# w.r.t. struct fields, so `forward()` above cannot be a primitive. To overcome it, we introduce
# a helper function batchnorm2d() which pretends to be a normal primitive but:
#  * takes a @nodiff struct
#  * separately, takes learnable gamma and beta parameters
function batchnorm2d(m::BatchNorm2d, gamma, beta, x)
    if m.training
        dims = (1, 2, 4)
        mu = batch_mean(x, dims)
        sigma2 = batch_var(x, dims)
        # track running mean and variance for each channel separately
        # based on: https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py#L50-L69
        # note: running stats aren't used during training, but are required for evaluation
        m.count += 1
        exp_avg_factor = eltype(x)(1.0) / m.momentum
        m.running_mean .= @. exp_avg_factor * mu + (1 - exp_avg_factor) * m.running_mean
        m.running_var .= @. exp_avg_factor * sigma2 * m.count / (m.count - 1) +
            (1 - exp_avg_factor) * m.running_var
        return batchnorm_impl(gamma, beta, x, mu, sigma2, m.momentum;
                         eps=m.eps, training=true)
    else
        return batchnorm_impl(gamma, beta, x, m.running_mean, m.running_var, m.momentum;
                         eps=m.eps, training=false)
    end
end


(m::BatchNorm2d)(x) = batchnorm2d(m, m.gamma, m.beta, x)


function ∇batchnorm2d(dy, m::BatchNorm2d, gamma, beta, x)
    if m.training
        dims = (1, 2, 4)
        return ∇batchnorm_impl(gamma, beta, x, dy, batch_mean(x, dims), batch_var(x, dims), m.momentum;
                          eps=m.eps, training=true)
    else
        return ∇batchnorm_impl(gamma, beta, x, dy, m.running_mean, m.running_var, m.momentum;
                          eps=m.eps, training=false)
    end
end


function register_batchnorm_derivs()
    # @nodiff batchnorm2d(_m, _gamma, _beta, x) _m
    # @diffrule batchnorm2d(_m, _gamma, _beta, x) _gamma getindex(∇batchnorm2d(dy, _m, _gamma, _beta, x), 1)
    # @diffrule batchnorm2d(_m, _gamma, _beta, x) _beta getindex(∇batchnorm2d(dy, _m, _gamma, _beta, x), 2)
    # @diffrule batchnorm2d(_m, _gamma, _beta, x) x getindex(∇batchnorm2d(dy, _m, _gamma, _beta, x), 3)
end





function forward(m::BatchNorm2d, x)
    size(x, ndims(x)-1) == length(BN.β) ||
        error("BatchNorm expected $(length(BN.β)) channels, got $(size(x, ndims(x)-1))")
    dims = length(size(x))
    channels = size(x, dims-1)
    affine_shape = ntuple(i->i == ndims(x) - 1 ? size(x, i) : 1, ndims(x))
    m = div(prod(size(x)), channels)
    γ = reshape(BN.γ, affine_shape...)
    β = reshape(BN.β, affine_shape...)
    if !istraining()
        μ = reshape(BN.μ, affine_shape...)
        σ² = reshape(BN.σ², affine_shape...)
        ϵ = BN.ϵ
    else
        T = eltype(x)
        axes = [1:dims-2; dims] # axes to reduce along (all but channels axis)
        μ = mean(x, dims = axes)
        σ² = sum((x .- μ) .^ 2, dims = axes) ./ m
        ϵ = convert(T, BN.ϵ)
        # update moving mean/std
        mtm = BN.momentum
        S = eltype(BN.μ)
        BN.μ  = (1 - mtm) .* BN.μ .+ mtm .* S.(reshape(μ, :))
        BN.σ² = (1 - mtm) .* BN.σ² .+ (mtm * m / (m - 1)) .* S.(reshape(σ², :))
    end

    let λ = BN.λ
        x̂ = (x .- μ) ./ sqrt.(σ² .+ ϵ)
        λ.(γ .* x̂ .+ β)
    end
end
