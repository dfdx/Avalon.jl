logistic(x::Real) = one(x) / (one(x) + exp(-x))
∇logistic(dy, x) = logistic(x) * (one(x) - logistic(x)) * dy
const sigmoid = logistic
# + culogistic in cuda.jl

softplus(x::Real) = log(exp(x) + one(x))
∇softplus(dy, x) = logistic(x) * dy
# + cusoftplus in cuda.jl

softsign(x::Real) = x / (one(x) + abs(x))
∇softsign(dy, x::Real) = dy * one(x) / ((one(x) + abs(x)) ^ 2)

logsigmoid(x::Real) = -softplus(-x)
∇logsigmoid(dy, x::Real) = one(x) / (exp(x) + one(x))

relu(x::Real) = max(zero(x), x)
∇relu(dy::Real, y::Real) = ifelse(y > 0, dy, zero(y))

leakyrelu(x::T, alpha) where T <: Real = max(T(alpha) * x, x)
∇leakyrelu(dy::Real, y::T, alpha) where T <: Real = ifelse(y > 0, dy, T(alpha))

elu(x::Real, alpha) = NNlib.elu(x, alpha)
∇elu(dy, x::Real, alpha) = ifelse(x >= 0, one(x), alpha * exp(x))
# + ∇cuelu in cuda.jl

softmax(x::AbstractArray) = NNlib.softmax(x)
∇softmax(dy, x) = NNlib.∇softmax(dy, x)

logsoftmax(x::AbstractArray) = NNlib.logsoftmax(x)
∇logsoftmax(dy, x) = NNlib.∇logsoftmax(dy, x)


function register_activation_derivs()
    @diffrule logistic(x::Real) x ∇logistic(dy, x)
    @diffrule softplus(x::Real) x ∇softplus(dy, x)
    @diffrule softsign(x::Real) x ∇softsign(dy, x)
    @diffrule logsigmoid(x::Real) x ∇logsigmoid(dy, x)
    @diffrule relu(x::Real) x ∇relu(dy, y)
    @diffrule leakyrelu(x::Real, _alpha::Real) x ∇leakyrelu(dy, x, _alpha)
    @diffrule elu(x::Real, _alpha::Real) x ∇elu(dy, x, _alpha)
    @nodiff leakyrelu(x::Real, _alpha::Real) _alpha
    
    @diffrule softmax(x) x ∇softmax(dy, x)
    @diffrule logsoftmax(x) x ∇logsoftmax(dy, x)
end
