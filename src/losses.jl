################################################################################
#                          Negative log-likelihood                             #
################################################################################

# NOTE 1: This NLL implementation has been cross-checked using Zygote,
# but there are no tests in the repo since gradcheck() doesn't support integer arguments.
# Thus if changed, new implementation should be checked with the current one.

# NOTE 2: Indexing in CuArrays is very, very slow, but I couldn't find a way
# to avoid it. One option to try is to use sum(X .* mask) where mask contains
# 1.0s on true-class positions for each observation and 0.0s elsewhere.
# This mask can be precalculated during forward pass and re-used during reverse pass.
# Another option is to calculate NLL on CPU and then copy it back to GPU,
# but in this case any benchmarks may vary across different GPUs.

"""
Negative log-likelihood. ŷ should be a vector of normalized log probabilities.
"""
function nllloss(ŷ::AbstractMatrix, c::AbstractVector{<:Real})
    loss = 0
    for j=1:size(ŷ, 2)
        i = Int(c[j])
        loss += -ŷ[i, j]
    end
    return loss / length(c)
end

function ∇nllloss(ds::Real, ŷ::AbstractMatrix, c::AbstractVector{<:Real})
    dŷ = zero(ŷ)
    len = length(c)
    # assuming instances are on columns, we can make it configurable later
    for j=1:size(ŷ, 2)
        i = Int(c[j])
        dŷ[i, j] = -ds / length(c)
    end
    return dŷ
end


"""
Negative log-likelihood.

This method takes list of classes `c` as one-hot encoded matrix, i.e. each column contains
exactly one 1.0 at position corresponding to true class
"""
function nllloss(ŷ::AbstractMatrix, y::AbstractMatrix{<:Real})
    return -sum(ŷ .* y) / size(ŷ, 2)
end

function ∇nllloss(dy::Real, ŷ::AbstractMatrix, y::AbstractMatrix{<:Real})
    p = ŷ .* y
    s = sum(p)
    scale = -1 / size(ŷ, 2)
    ∇s = Yota.sum_grad(p, scale)
    ∇ŷ = Yota.unbroadcast_prod_x(ŷ, y, ∇s)
    # ∇y = Yota.unbroadcast_prod_y(ŷ, y, ∇s)
    return ∇ŷ
end


# another implementation for future reference
function nllloss2(logp::AbstractMatrix, c::Vector{<:Real})
   return -LinearAlgebra.tr(logp[c, :]) / length(c)
end


################################################################################
#                               Cross Entropy                                  #
################################################################################

crossentropyloss(x::AbstractMatrix, c::Union{AbstractMatrix{<:Real}, AbstractVector{<:Real}}) =
    nllloss(logsoftmax(x), c)


################################################################################
#                                 Mean Squared Loss                            #
################################################################################

mseloss(inp::AbstractArray, target::AbstractArray) = mean((inp .- target) .^ 2)


function register_loss_derivs()
    # @diffrule nllloss(x, _c) x ∇nllloss(dy, x, _c)
    # @nodiff nllloss(x, _c) _c
end
