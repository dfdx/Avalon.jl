###############################################################################
#                                 Embedding                                   #
###############################################################################

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


# function embedding(W::AbstractMatrix, s::AbstractMatrix)
#     E = similar(W, size(W, 1), size(s)...)
#     @inbounds for j=1:size(s, 2)
#         E[:, :, j] = W[:, s[:, j]]
#         # e = reshape(e, size(e)..., 1)
#         # push!(es, e)
#     end
#     # return cat(es..., dims=3)
#     return E
# end


function embedding(W::AbstractMatrix, s::AbstractMatrix)
    E = similar(W, size(W, 1), size(s)...)
    Yota.Scatter.gather(E, W, s)
    return E
end


function ∇embedding_W(dy::AbstractArray{T, 3}, W::AbstractMatrix, s::AbstractMatrix) where T
    dW = zero(W)
    for j=1:size(s, 2)
        Yota.ungetindex!(dW, W, dy[:, :, j], :, s[:, j])        
    end
    return dW
end


(l::Embedding)(s::AbstractMatrix) = embedding(l.W, s)


function register_embed_derivs()
    @diffrule embedding(_W, _s) _W ∇embedding_W(dy, _W, _s)
    @nodiff embedding(_W, _s) _s
end
