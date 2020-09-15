################################################################################
#                               Linear                                         #
################################################################################

mutable struct Linear
    W::AbstractMatrix{T} where T
    b::AbstractVector{T} where T
end

function Linear(in_features::Int, out_features::Int)
    k_sqrt = sqrt(1 / in_features)
    d = Uniform(-k_sqrt, k_sqrt)
    return Linear(rand(d, out_features, in_features), rand(d, out_features))
end
Linear(in_out::Pair{Int, Int}) = Linear(in_out[1], in_out[2])

function Base.show(io::IO, l::Linear)
    o, i = size(l.W)
    print(io, "Linear($i=>$o)")
end

(l::Linear)(x::Union{AbstractVector{T}, AbstractMatrix{T}}) where T = l.W * x .+ l.b


################################################################################
#                               Sequential                                     #
################################################################################

mutable struct Sequential
    seq::Tuple
end

Sequential(args...) = Sequential(args)

function Base.show(io::IO, s::Sequential)
    println(io, "Sequential(")
    for m in s.seq
        println(io, "  $m,")
    end
    print(io, ")")
end


function (s::Sequential)(x)
    for m in s.seq
        x = m(x)
    end
    return x
end


################################################################################
#                           Convolutions                                       #
################################################################################

# Although internall 1D, 2D and 3D convolutions use the same convNd method,
# we keep different wrappers to make API easier to understand

################################# Conv1D #######################################

"""
1D convolutional layer.

Input and output dimensions: (sequence_length, channels, batch_size)
"""
mutable struct Conv1d
    W::AbstractArray{T,3} where T
    b::Union{AbstractVector{T} where T, Nothing}
    stride
    padding
    dilation
end

function Conv1d(in_channels::Int, out_channels::Int, kernel_size::Int;
                stride=1, padding=0, dilation=1, bias=true)
    kernel_tuple = (kernel_size,)
    # init weights same as in https://pytorch.org/docs/stable/nn.html#conv2d
    k_sqrt = sqrt(1 / (in_channels * prod(kernel_tuple)))
    d = Uniform(-k_sqrt, k_sqrt)
    W = rand(d, kernel_tuple..., in_channels, out_channels)
    b = bias ? rand(d, out_channels) : nothing
    return Conv1d(W, b, stride, padding, dilation)
end
Conv1d(in_out::Pair{Int, Int}, k) = Conv1d(in_out[1], in_out[2], k)


function Base.show(io::IO, c::Conv1d)
    k1, i, o = size(c.W)
    print(io, "Conv1d($i=>$o, $(k1))")
end

function (c::Conv1d)(x::AbstractArray{T,3}) where T
    y = convNd(x, c.W; stride=c.stride, padding=c.padding, dilation=c.dilation)
    if c.b != nothing
        y = y .+ reshape(c.b, (1, length(c.b), 1))  # TODO: check for 1D and 3D!
    end
    return y
end


################################# Conv2D #######################################

mutable struct Conv2d
    W::AbstractArray{T,4} where T
    b::Union{AbstractVector{T} where T, Nothing}
    stride
    padding
    dilation
end

function Conv2d(in_channels::Int, out_channels::Int, kernel_size::Union{Int, NTuple{2, Int}};
                stride=1, padding=0, dilation=1, bias=true)
    kernel_tuple = kernel_size isa Tuple ? kernel_size : (kernel_size, kernel_size)
    # init weights same as in https://pytorch.org/docs/stable/nn.html#conv2d
    k_sqrt = sqrt(1 / (in_channels * prod(kernel_tuple)))
    d = Uniform(-k_sqrt, k_sqrt)
    W = rand(d, kernel_tuple..., in_channels, out_channels)
    b = bias ? rand(d, out_channels) : nothing
    return Conv2d(W, b, stride, padding, dilation)
end
Conv2d(in_out::Pair{Int, Int}, k) = Conv2d(in_out[1], in_out[2], k)

function Base.show(io::IO, c::Conv2d)
    k1, k2, i, o = size(c.W)
    print(io, "Conv2d($i=>$o, $(k1)x$(k2))")
end

function (c::Conv2d)(x::AbstractArray{T,4}) where T
    y = convNd(x, c.W; stride=c.stride, padding=c.padding, dilation=c.dilation)
    if c.b != nothing
        y = y .+ reshape(c.b, (1, 1, length(c.b), 1))
    end
    return y
end


################################# Conv3D #######################################

mutable struct Conv3d
    W::AbstractArray{T,5} where T
    b::Union{AbstractVector{T} where T, Nothing}
    stride
    padding
    dilation
end

function Conv3d(in_channels::Int, out_channels::Int, kernel_size::Union{Int, NTuple{2, Int}};
                stride=1, padding=0, dilation=1, bias=true)
    kernel_tuple = kernel_size isa Tuple ? kernel_size : (kernel_size, kernel_size, kernel_size)
    # init weights same as in https://pytorch.org/docs/stable/nn.html#conv2d
    k_sqrt = sqrt(1 / (in_channels * prod(kernel_tuple)))
    d = Uniform(-k_sqrt, k_sqrt)
    W = rand(d, kernel_tuple..., in_channels, out_channels)
    b = bias ? rand(d, out_channels) : nothing
    return Conv3d(W, b, stride, padding, dilation)
end
Conv3d(in_out::Pair{Int, Int}, k) = Conv3d(in_out[1], in_out[2], k)

function Base.show(io::IO, c::Conv3d)
    k1, k2, k3, i, o = size(c.W)
    print(io, "Conv3d($i=>$o, $(k1)x$(k2)x$(k3))")
end

function (c::Conv3d)(x::AbstractArray{T,5}) where T
    y = convNd(x, c.W; stride=c.stride, padding=c.padding, dilation=c.dilation)
    if c.b != nothing
        y = y .+ reshape(c.b, (1, 1, length(c.b), 1))
    end
    return y
end


################################################################################
#                           Loss Functions                                     #
################################################################################

mutable struct NLLLoss
end

Base.show(io::IO, loss::NLLLoss) = print(io, "NLLLoss()")

(loss::NLLLoss)(logp::AbstractMatrix, c::Union{AbstractMatrix{<:Real}, AbstractVector{<:Real}}) =
    nllloss(logp, c)


mutable struct CrossEntropyLoss
end

Base.show(io::IO, loss::CrossEntropyLoss) = print(io, "CrossEntropyLoss()")

(loss::CrossEntropyLoss)(x::AbstractMatrix, c::Union{AbstractMatrix{<:Real}, AbstractVector{<:Real}}) =
    crossentropyloss(x, c)


mutable struct MSELoss
end

Base.show(io::IO, loss::MSELoss) = print(io, "MSELoss()")


(loss::MSELoss)(x::AbstractArray, x_target::AbstractArray) = mseloss(x, x_target)


################################################################################
#                                 Common                                       #
################################################################################


function trainmode!(m, train::Bool=true)
    # by default, recursively call trainmode!() on all fields
    T = typeof(m)
    for fld in fieldnames(T)
        trainmode!(getfield(m, fld), train)
    end
end
testmode!(m) = trainmode!(m, false)
