# using CuArrays
# using CUDAnative
import CUDA.CUDNN

# CuArrays doesn't provide a dedicated version of conv() for CuArrays, and so call to
# NNlib.conv(CuArray(), CuArray()) leads to CPU-based algorithm on GPU arrays, which is terribly slow.
# Here we implement our wrapper conv2d() for CuArrays via in-place version CuArrays.conv!().
# The same applies to maxpool() which we implement using maxpool!().
# Note that later we will have in-place rules and rewrite them on the tape, but we still
# need to make all operations works without any reference to gradients.

## conv2d

function conv2d(x::CuArray{T,4}, w::CuArray{T,4}; stride=1, padding=0, dilation=1) where T
    cdims = DenseConvDims(x, w; stride=stride, padding=padding, dilation=dilation)
    y = similar(x, NNlib.output_size(cdims)..., NNlib.channels_out(cdims), size(x, 4))
    return CUDNN.conv!(y, x, w, cdims)
end

function ∇conv2d_w(dy::CuArray{T,4}, x::CuArray{T,4}, w::CuArray{T,4}; stride=1, padding=0, dilation=1) where T
    cdims = DenseConvDims(x, w; stride=stride, padding=padding, dilation=dilation)
    dw = similar(w)
    return CUDNN.∇conv_filter!(dw, x, dy, cdims)
end

function ∇conv2d_x(dy::CuArray{T,4}, x::CuArray{T,4}, w::CuArray{T,4}; stride=1, padding=0, dilation=1) where T
    cdims = DenseConvDims(x, w; stride=stride, padding=padding, dilation=dilation)
    dx = similar(x)
    return CUDNN.∇conv_data!(dx, dy, w, cdims)
end


## maxpool2d

function maxpool2d(x::CuArray, kernel_size; stride=kernel_size, padding=0, dilation=1)
    pdims = PoolDims(x, kernel_size; stride=stride, padding=padding, dilation=dilation)
    y = similar(x, NNlib.output_size(pdims)..., NNlib.channels_out(pdims), size(x, 4))
    return CUDNN.maxpool!(y, x, pdims)
end

function ∇maxpool2d_x(dy::CuArray, y::CuArray, x::CuArray, kernel_size; stride=kernel_size, padding=0, dilation=1)
    pdims = PoolDims(x, kernel_size; stride=stride, padding=padding, dilation=dilation)
    dx = similar(x)
    return NNlib.∇maxpool!(dx, dy, y, x, pdims)
end


## activations

culogistic(x) = one(x) / (one(x) + CUDA.exp(-x))
CUDA.cufunction(::typeof(logistic)) = culogistic

∇culogistic(dy, x) = culogistic(x) * (one(x) - culogistic(x)) * dy
CUDA.cufunction(::typeof(∇logistic)) = ∇culogistic

cusoftplus(x) = CUDA.log(CUDA.exp(x) + one(x))
CUDA.cufunction(::typeof(softplus)) = cusoftplus

∇cusoftplus(dy, x) = culogistic(x) * dy
CUDA.cufunction(::typeof(∇softplus)) = ∇cusoftplus

∇cuelu(dy, x::Real, alpha) = ifelse(x >= 0, x/1, alpha * CUDA.exp(x))
CUDA.cufunction(::typeof(∇elu)) = ∇cuelu


## batchnorm

function batchnorm_impl(gamma::CuArray, beta::CuArray, x::CuArray,
                        mu::CuArray, sigma2::CuArray, momentum::Real; eps, training)
    CUDNN.batchnorm(gamma, beta, x, mu, sigma2, momentum; eps=eps, training=training)
end

function ∇batchnorm_impl(gamma::CuArray, beta::CuArray, x::CuArray, dy::CuArray,
                         mu::CuArray, sigma2::CuArray, momentum; eps, training)
    CUDNN.∇batchnorm(gamma, beta, x, dy, mu, sigma2, momentum; eps=eps, training=training)
end


