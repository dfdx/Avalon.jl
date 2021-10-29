import NNlib: DenseConvDims, PoolDims

## generic convolution

function convNd(x, w; stride=1, padding=0, dilation=1)
    # TODO: unflipweights
    cdims = DenseConvDims(x, w; stride=stride, padding=padding, dilation=dilation)
    return NNlib.conv(x, w, cdims)
end

function rrule(::typeof(convNd), x, w; stride=1, padding=0, dilation=1)
    cdims = DenseConvDims(x, w; stride=stride, padding=padding, dilation=dilation)
    y = NNlib.conv(x, w, cdims)
    function convNd_pullback(dy)
        # TODO: unflipweights
        dy = unthunk(dy)
        return NoTangent(), NNlib.∇conv_data(dy, w, cdims), NNlib.∇conv_filter(x, dy, cdims)
    end
    return y, convNd_pullback
end

## specific convolution functions

const conv1d = convNd
const conv2d = convNd
const conv3d = convNd


## maxpool2d

function maxpool2d(x, kernel_size; stride=kernel_size, padding=0, dilation=1)
    pdims = PoolDims(x, kernel_size; stride=stride, padding=padding, dilation=dilation)
    return NNlib.maxpool(x, pdims)
end

function rrule(::typeof(maxpool2d), x, kernel_size; stride=kernel_size, padding=0, dilation=1)
    pdims = PoolDims(x, kernel_size; stride=stride, padding=padding, dilation=dilation)
    y = NNlib.maxpool(x, pdims)
    function maxpool_pullback(dy)
        dy = unthunk(dy)
        return NoTangent(), NNlib.∇maxpool(dy, y, x, pdims), NoTangent()
    end
    return y, maxpool_pullback
end