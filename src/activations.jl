import ChainRulesCore
import ChainRulesCore: rrule, @scalar_rule


const sigmoid = NNlib.sigmoid
const logistic = NNlib.sigmoid
const logsigmoid = NNlib.logsigmoid
const softplus = NNlib.softplus
const softsign = NNlib.softsign
const relu = NNlib.relu
const leakyrelu = NNlib.leakyrelu
const elu = NNlib.elu
const softmax = NNlib.softmax
const logsoftmax = NNlib.logsoftmax


# extend rrules defined in NNlib
# note that Zygote handles these cases implicitely, while we define
# explicit rrules for broadcasting them
UNARY_ACTS = [ # f, df
    (:logsigmoid,     :(1 / (exp(x) + 1))),
    (:softsign,       :(one(x) / ((one(x) + abs(x)) ^ 2))),
]

for (f, df) in UNARY_ACTS
    @eval @scalar_rule($f(x), $df)

    pullback = Symbol(:broadcasted_, f, :_pullback)
    @eval function ChainRulesCore.rrule(::typeof(Broadcast.broadcasted),
                         ::typeof($f), x::AbstractArray)
        Ω = $f.(x)
        function $pullback(Δ)
            NoTangent(), NoTangent(), @.(Δ * $df)
        end
        return Ω, $pullback
    end
end


∇leakyrelu(dy::Real, y::T, alpha) where T <: Real = ifelse(y > 0, T(dy), T(alpha))

function ChainRulesCore.rrule(::typeof(leakyrelu), x::Number, alpha::Number)
    y = leakyrelu(x, alpha)
    leakyrelu_pullback(dy) = (NoTangent(), ∇leakyrelu(unthunk(dy), x, alpha), NoTangent())
    return y, leakyrelu_pullback
end

function ChainRulesCore.rrule(::typeof(Broadcast.broadcasted), ::typeof(leakyrelu),
        x::AbstractArray{T}, alpha::Number) where T
    alpha = T(alpha)
    y = leakyrelu.(x, alpha)
    function bcast_leakyrelu_pullback(dy)
        dy = unthunk(dy)
        return (NoTangent(), NoTangent(), ∇leakyrelu.(dy, y, alpha), NoTangent())
    end
    return y, bcast_leakyrelu_pullback
end