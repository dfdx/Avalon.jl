## recurrent networks
## LSTM implementation is taken from:
## https://github.com/FluxML/Flux.jl/blob/7104fd933200833d3d73e6dec2f9572e3436f4dc/src/layers/recurrent.jl


################################################################################
#                                 Vanilla RNN                                  #
################################################################################

mutable struct RNNCell
  W_ih::AbstractMatrix
  W_hh::AbstractMatrix
  b_ih::AbstractVector
  b_hh::AbstractVector
end


function RNNCell(inp::Integer, hid::Integer)
    k_sqrt = sqrt(1 / hid)
    d = Uniform(-k_sqrt, k_sqrt)
    return RNNCell(rand(d, hid, inp), rand(d, hid, hid), rand(d, hid), rand(hid))
end

Base.show(io::IO, m::RNNCell) = print(io, "RNNCell($(size(m.W_ih, 2)) => $(size(m.W_ih, 1)))")


# using proxy function to be able to compute numeric gradient in tests
function rnn_forward(W_ih, W_hh, b_ih, b_hh, x, h)
    inp_v = W_ih * x .+ b_ih
    hid_v = W_hh * h .+ b_hh
    h_ = tanh.(inp_v .+ hid_v)
    return h_
end


# input should be of size (inp_size, batch)
function forward(m::RNNCell, x::AbstractMatrix, h::AbstractMatrix)
    rnn_forward(m.W_ih, m.W_hh, m.b_ih, m.b_hh, x, h)
end

(m::RNNCell)(x::AbstractMatrix, h::AbstractMatrix) = forward(m, x, h)


mutable struct RNN
    cell::RNNCell
end
RNN(inp_size, hid_size) = RNN(RNNCell(inp_size, hid_size))
RNN(inp_hid::Pair{Int}) = RNN(RNNCell(inp_hid[1], inp_hid[2]))


function init_hidden(m::RNN, batch_size::Integer)
    hid_size = length(m.cell.b_hh)
    return zeros(hid_size, batch_size)
end


# x_seq should be of size (inp_size, batch, seq_len)
# h should be of size (hid_size, batch)
function forward(m::RNN, x_seq::AbstractArray{T, 3}, h::AbstractArray{T, 2}) where T
    # device = Yota.guess_device([h])
    inp_size, batch, seq_len = size(x_seq)
    hid_size = length(m.cell.b_hh)
    h_all_sz = (size(h)..., 1)
    cell = m.cell
    # 1st element
    x = x_seq[:, :, 1]
    h = forward(cell, x, h)
    h_all = reshape(h, h_all_sz)
    # all other elements
    for i=2:size(x_seq, 3)
        x = x_seq[:, :, i]
        h = forward(cell, x, h)
        h_all = cat(h_all, reshape(h, h_all_sz); dims=3)
    end
    return h_all, h
end

(m::RNN)(x::AbstractArray{T,3}, h::AbstractArray{T,2}) where T = forward(m, x, h)


################################################################################
#                                  LSTM                                        #
################################################################################

mutable struct LSTMCell
  W_ih::AbstractMatrix
  W_hh::AbstractMatrix
  b_ih::AbstractVector
  b_hh::AbstractVector
end


function LSTMCell(inp::Integer, hid::Integer)
    k_sqrt = sqrt(1 / hid)
    d = Uniform(-k_sqrt, k_sqrt)
    return LSTMCell(rand(d, 4*hid, inp), rand(d, 4*hid, hid), rand(d, 4*hid), rand(4*hid))
end
LSTMCell(inp_hid::Pair{Int}) = LSTMCell(inp_hid[1], inp_hid[2])

Base.show(io::IO, m::LSTMCell) = print(io, "LSTMCell($(size(m.W_ih, 2)) => $(size(m.W_ih, 1)))")



slice(len, idx) = (1:len) .+ len*(idx-1)


function lstm_forward(W_ih, W_hh, b_ih, b_hh, x, h, c)
    hid_len = size(h, 1)
    σ = tanh
    # weight breakdown in PyTorch is described here:
    # https://github.com/pytorch/pytorch/blob/0c48092b2270d56cdab327bd1ff0ca89f5b7d569/torch/nn/modules/rnn.py#L479-L487
    # however, we don't split weights into W_ii, W_if, etc., but instead first calculate
    # all linear transformations as `y` and only after that take slices of this variable
    # to calculate i, f, g and o
    y = W_ih*x .+ b_ih .+ W_hh*h .+ b_hh
    i = σ.(y[slice(hid_len, 1), :])
    f = σ.(y[slice(hid_len, 2), :])
    g = tanh.(y[slice(hid_len, 3), :])
    o = σ.(y[slice(hid_len, 4), :])
    c_ = f .* c .+ i .* g
    h_ = o .* tanh.(c_)
    return h_, c_
end

# x should be of size (inp_size, batch_size)
# h and c should be of size (hid_size, batch_size)
function forward(m::LSTMCell, x::AbstractMatrix, h::AbstractMatrix, c::AbstractMatrix)
    return lstm_forward(m.W_ih, m.W_hh, m.b_ih, m.b_hh, x, h, c)
end

(m::LSTMCell)(x::AbstractMatrix, h::AbstractMatrix, c::AbstractMatrix) = forward(m, x, h, c)


mutable struct LSTM
    cell::LSTMCell
end
LSTM(inp_size, hid_size) = LSTM(LSTMCell(inp_size, hid_size))
LSTM(inp_hid::Pair{Int}) = LSTM(LSTMCell(inp_hid[1], inp_hid[2]))


function init_hidden(m::LSTM, batch_size::Integer)
    hid_size = length(m.cell.b_hh) ÷ 4
    return zeros(hid_size, batch_size), zeros(hid_size, batch_size)
end


# x_seq should be of size (inp_size, batch, seq_len)
# h and c should be of size (hid_size, batch)
function forward(m::LSTM, x_seq::AbstractArray{T, 3}, h::AbstractArray{T, 2}, c::AbstractArray{T, 2}) where T
    # device = Yota.guess_device([h])
    inp_size, batch, seq_len = size(x_seq)
    hid_size = size(h, 1)
    h_all_sz = (size(h)..., 1)
    cell = m.cell
    # 1st element
    x = x_seq[:, :, 1]
    h, c = forward(cell, x, h, c)
    h_all = reshape(h, h_all_sz)
    # all other elements
    for i=2:size(x_seq, 3)
        x = x_seq[:, :, i]
        h, c = forward(cell, x, h, c)
        h_all = cat(h_all, reshape(h, h_all_sz); dims=3)
    end
    return h_all, h, c
end

(m::LSTM)(x::AbstractArray{T,3}, h::AbstractArray{T,2}, c::AbstractArray{T,2}) where T = forward(m, x, h, c)


################################################################################
#                                  GRU                                         #
################################################################################

mutable struct GRUCell
  W_ih::AbstractMatrix
  W_hh::AbstractMatrix
  b_ih::AbstractVector
  b_hh::AbstractVector
end


function GRUCell(inp::Integer, hid::Integer)
    k_sqrt = sqrt(1 / hid)
    d = Uniform(-k_sqrt, k_sqrt)
    return GRUCell(rand(d, 3*hid, inp), rand(d, 3*hid, hid), rand(d, 3*hid), rand(3*hid))
end
GRUCell(inp_hid::Pair{Int}) = GRUCell(inp_hid[1], inp_hid[2])

Base.show(io::IO, m::GRUCell) = print(io, "GRUCell($(size(m.W_ih, 2)) => $(size(m.W_ih, 1)))")


function gru_forward(W_ih, W_hh, b_ih, b_hh, x, h)
    hid_len = size(h, 1)
    σ = tanh
    y_i = W_ih*x .+ b_ih
    y_h = W_hh*h .+ b_hh
    # parameters are slices of precalculated y_i and y_h
    s1 = slice(hid_len, 1); s2 = slice(hid_len, 2); s3 = slice(hid_len, 3)
    r = σ.(y_i[s1, :] .+ y_h[s1, :])
    z = σ.(y_i[s2, :] .+ y_h[s2, :])
    n = tanh.(y_i[s3, :] .+ r .* y_h[s3, :])
    h_ = (1 .- z) .* n .+ z .* h
    return h_
end

# x should be of size (inp_size, batch_size)  -- check
# h and c should be of size (hid_size, batch_size)  -- check
function forward(m::GRUCell, x::AbstractMatrix, h::AbstractMatrix)
    return gru_forward(m.W_ih, m.W_hh, m.b_ih, m.b_hh, x, h)
end

(m::GRUCell)(x::AbstractMatrix, h::AbstractMatrix) = forward(m, x, h)


mutable struct GRU
    cell::GRUCell
end
GRU(inp_size, hid_size) = GRU(GRUCell(inp_size, hid_size))
GRU(inp_hid::Pair{Int}) = GRU(GRUCell(inp_hid[1], inp_hid[2]))


function init_hidden(m::GRU, batch_size::Integer)
    hid_size = length(m.cell.b_hh) ÷ 3
    return zeros(hid_size, batch_size)
end


# x_seq should be of size (inp_size, batch, seq_len)
# h should be of size (hid_size, batch)
function forward(m::GRU, x_seq::AbstractArray{T, 3}, h::AbstractArray{T, 2}) where T
    # device = Yota.guess_device([h])
    inp_size, batch, seq_len = size(x_seq)
    hid_size = size(h, 1)
    h_all_sz = (size(h)..., 1)
    cell = m.cell
    # 1st element
    x = x_seq[:, :, 1]
    h = forward(cell, x, h)
    h_all = reshape(h, h_all_sz)
    for i=2:size(x_seq, 3)
        x = x_seq[:, :, i]
        h = forward(cell, x, h)
        h_all = cat(h_all, reshape(h, h_all_sz); dims=3)
    end
    return h_all, h
end

(m::GRU)(x::AbstractArray{T,3}, h::AbstractArray{T,2}) where T = forward(m, x, h)



## SAVING THIS CODE FOR FUTURE
## PyTorch provides optionally multilayer and biderectional RNN implementation.
## There are many ways to do the same thing, but to align the code with PyTorch
## I need to clearly understand their implementation, which is out of scope for me now.
## Thus I'm just saving this in-progress piece of code for future and stick
## with simple implementation for now.
## It's worth to note that at least multiple layers can be achieved on user level
## by stacking several RNNs

## mutable struct RNN
##     num_layers::Int
##     num_directions::Int
##     cells::RNNCell
## end

## RNN(inp::Integer, hid::Integer; num_layers::Int=1, biderectional::Bool=false) =
##     RNN(num_layers, biderectional ? 2 : 1, RNNCell(inp, hid))
## RNN(inp_hid::Pair{Int}; num_layers::Int=1, biderectional::Bool=false) =
##     RNN(inp_hid[1], inp_hid[2]; num_layers=num_layers, biderectional=biderectional)

## Base.show(io::IO, m::RNN) = print(io, "RNN($(m.cell)")

## function init_hidden(m::RNN, batch_size::Integer)
##     hid_size = length(m.cell.b_hh)
##     return zeros(hid_size, batch_size, m.num_layers * m.num_directions)
## end


## # x_seq should be of size (inp_size, batch_size, seq_len)
## # H should be of size (hid_size, batch_size, num_layers*num_directions)
## function forward(m::RNN, x_seq::AbstractArray{T, 3}, H::AbstractArray{T, 3}) where T
##     inp_size, batch_size, seq_len = size(x_seq)
##     hid_size = length(m.cell.b_hh)
##     # h_all = zeros(hid_size, batch_size, m.num_layers * m.num_directions)
##     # h_all_sz = (size(h)..., 1)
##     cell = m.cell
##     for l=1:m.num_layers
##         h = H[:, :, 2l - 1]           # l-th layer, forward direction
##         for i=1:size(x_seq, 3)
##             x = x_seq[:, :, i]
##             h = forward(cell, x, h)
##             # h_all = cat(h_all, reshape(h, h_all_sz); dims=3)
##         end
##         h = H[:, :, 2l]  # l-th layer, reverse direction
##         for i=size(x_seq, 3):1
##             x = x_seq[:, :, i]
##             h = forward(cell, x, h)
##             # h_all = cat(h_all, reshape(h, h_all_sz); dims=3)
##         end
##         # TODO: add another direction
##     end
##     # return h_all, h
##     return h
## end

## (m::RNN)(x::AbstractArray{T,3}, h::AbstractArray{T,2}) where T = forward(m, x, h)
