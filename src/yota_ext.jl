## temporary place for all the stuff that should go to Yota.jl

function Base.reshape(dy::ZeroTangent, ::Tuple)
    return dy
end

# import ChainRulesCore

# function ChainRulesCore.rrule(::Colon, a::Int, b::Int)
#     y = a:b
#     function colon_pullback(dy)
#         return NoTangent(), NoTangent(), NoTangent()
#     end
#     return y, colon_pullback
# end

# test_rrule(Colon(), 1, 2; ouput_tangent=o)