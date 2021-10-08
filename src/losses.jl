################################################################################
#                          Negative log-likelihood                             #
################################################################################

"""
Negative log-likelihood. ŷ should be a vector of normalized log probabilities.
"""
function nllloss(ŷ::AbstractMatrix, c::AbstractVector{<:Real})
    loss = @tullio (+) r = ŷ[c[k], k]
    return -loss / length(c)
end


function ∇nllloss(Δ::Real, ŷ::AbstractMatrix, c::AbstractVector{<:Real})
    dŷ = zero(ŷ)
    val = -Δ / length(c)
    @tullio dŷ[c[k], k] = val
end

function ChainRulesCore.rrule(::typeof(nllloss), ŷ::AbstractMatrix, c::AbstractVector{<:Real})
    y = nllloss(ŷ, c)
    function nllloss_pullback(Δ)
        return NoTangent(), ∇nllloss(unthunk(Δ), ŷ, c), NoTangent()
    end
    return y, nllloss_pullback
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