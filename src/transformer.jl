include("core.jl")


###############################################################################
#                                 Embedding                                   #
###############################################################################

# using Tullio, KernelAbstractions  # , LoopVectorization


# """
# onehot(s::AbstractArray)

# Create a dense one-hot encoded array from an array of labels.

# Given an array of N dimensions, onehot() returns array of N+1 dimensions,
# where first dimension is equal to the number of classes. For example,
# input matrix of size (seq_len, batch_size) will produce 3D array of size
# (n_classes, seq_len, batch_size).
# """
# function onehot(s::AbstractVector; n_classes=maximum(s))
#     x = similar(s, n_classes, length(s))
#     return @tullio x[i, j] = (i == s[j]) (i ∈ 1:n_classes)
# end

# # for matrices onehot() returns 3D array of shape(n_classes, seq_len, batch_size)
# function onehot(s::AbstractArray; n_classes=maximum(s))
#     return reshape(onehot(vec(s); n_classes=n_classes), :, size(s)...)
# end


mutable struct Embedding
    W::AbstractMatrix{T} where T
end


function Embedding(in_features::Int, out_features::Int)
    k_sqrt = sqrt(1 / in_features)
    d = Uniform(-k_sqrt, k_sqrt)
    return Embedding(rand(d, out_features, in_features))
end
Embedding(in_out::Pair{Int, Int}) = Embedding(in_out[1], in_out[2])

function Base.show(io::IO, l::Embedding)
    o, i = size(l.W)
    print(io, "Embedding($i=>$o)")
end


function embedding(W::AbstractMatrix, s::AbstractMatrix)
    es = []
    for j=1:size(s, 2)
        e = W[:, s[:, j]]
        e = reshape(e, size(e)..., 1)
        push!(es, e)
    end
    return cat(es..., dims=3)
end


function ∇embedding_W(dy::AbstractArray{T, 3}, W::AbstractMatrix, s::AbstractMatrix) where T
    dW = zero(W)
    for j=1:size(s, 2)
        Yota.ungetindex!(dW, W, dy[:, :, j], :, s[:, j])        
    end
    return dW
end


function register_embed_derivs()
    @diffrule embedding(_W, _s) _W ∇embedding_W(dy, _W, _s)
    @nodiff embedding(_W, _s) _s
end

(l::Embedding)(s::AbstractMatrix) = embedding(l.W, s)


################################################################################
#                                                                              #
################################################################################

# function unget!(dx::AbstractArray, dy, i...)
#     dx[i...] .+= dy
#     return dx
# end



# function countmap(y)
#     d = Dict{Int, Int}()
#     for val in y
#         if isnan(val)
#             continue
#         end
#         if val in keys(d)
#             d[val] += 1
#         else
#             d[val] = 1
#         end
#     end
#     return d
# end


# function reduce_indices(I)
#     cnt = countmap(I)
#     return collect(keys(cnt)), collect(values(cnt))
# end


# function unget!(dx, dy, I...)
#     # for (dy_ii, ii) in zip(dy, Iterators.product(I...))
#     for ii in Iterators.product(I...)
#         # @show ii
#         # dx[ii...] += dy_ii
#     end
#     return dx
# end



# import ScatterNNlib



# function my_scatter_add!(A::AbstractArray, v::AbstractArray, I...)
#     II = collect(Iterators.product(I...))
#     A2 = ScatterNNlib.scatter_add!(reshape(A, 1, size(A)...), reshape(v, 1, size(v)...), II)
#     return dropdims(A2, dims=1)
# end


function nomain()
    A = zeros(4)
    I = ([1, 3, 1],)
    v = [1, 1, 1]

    II = collect(Iterators.product(I...))
    dropdims(scatter_add!(reshape(A, 1, size(A)...), reshape(v, 1, size(v)...), II), dims=1)

    # partial solution
    A[I] .+= v


    A = zeros(4, 5) |> cu
    I = (1:size(A, 1), [1, 3, 1]) |> cu
    v = ones(4, 3) |> cu

    # solution
    II = collect(Iterators.product(I...)) |> cu
    dropdims(scatter_add!(reshape(A, 1, size(A)...), reshape(v, 1, size(v)...), II), dims=1)

end


function main()
#     x = rand(4) |> cu
#     dx = zeros(4) |> cu
#     s = [1, 3, 1] |> cu
#     y = x[s]
#     dy = ones(size(y)) |> cu
#     unget!(dx, dy, s)
#     # use Tullio here?
# #     @tullio (dx[i] += dy[i]) (i in s)
#     for i in s
#         dx[i] += dy[i]
#     end


#     I2, w = reduce_indices(I)
#     dx[I2...] .+= dy


    s = [1 4; 2 5; 1 6]
    # x = onehot(s)
    emb = Embedding(6 => 4)
    W = emb.W
    # W = reshape(emb.W, size(emb.W)..., 1)
    # W = cat(W, W, dims=3)
    # NNlib.batched_mul(W, x)

    f = (W, s) -> sum(embedding(W, s))
    _, y_grads = grad(f, W, s)
    y_grads[1]
    w_ngrad = ngradient2(f, (W, s), 1)




end
