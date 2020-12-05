## Code for optimizers adapted from
## https://github.com/jacobcvt12/GradDescent.jl/
import ChainRulesCore


abstract type Optimizer end


function Yota.update!(opt::Optimizer, m, gm; ignore=Set())
    # we use Zero() to designate values that need not to be updated
    gm isa ChainRulesCore.Zero && return
    for (path, gx) in Yota.path_value_pairs(gm)
        if !in(path, ignore) && !isa(gx, ChainRulesCore.Zero)
            x_t0 = Yota.getfield_nested(m, path)
            x_t1 = make_update!(opt, path, x_t0, gx)
            Yota.setfield_nested!(m, path, x_t1)
        end
    end
    opt.t += 1
end


function Yota.update!(opt::Optimizer, x::AbstractArray, gx; ignore=Set())
    x .= make_update!(opt, (), x, gx)
    opt.t += 1
end


################################################################################
#                                    SGD                                       #
################################################################################

mutable struct SGD <: Optimizer
    t::Int
    lr::Float32
    momentum::Float32
    v_t::Dict{Any, Any}   # path => previous velocity
end

function SGD(lr; momentum=0)
    # @assert momentum >= 0.0 "momentum must be >= 0"
    SGD(0, lr, momentum, Dict())
end

Base.show(io::IO, opt::SGD) = print(io, "SGD(lr=$(opt.lr), momentum=$(opt.momentum))")


function make_update!(opt::SGD, path, x, gx)
    v_t = get(opt.v_t, path, zero(gx))
    v_t1 = opt.momentum .* v_t .+ opt.lr .* gx
    opt.v_t[path] = v_t1
    x_t = x
    x_t1 = x_t .- v_t1
    return x_t1
end


################################################################################
#                                RMSprop                                       #
################################################################################


mutable struct RMSprop <: Optimizer
    t::Int32
    eps::Float32
    eta::Float32
    gamma::Float32
    E_g_sq_t::Dict{Any, Any}
end

function RMSprop(; eta::Real=0.001, gamma::Real=0.01, eps::Real=1e-8)
    @assert eta > 0.0 "eta must be greater than 0"
    @assert gamma > 0.0 "gamma must be greater than 0"
    @assert eps > 0.0 "eps must be greater than 0"

    RMSprop(0, eps, eta, gamma, Dict())
end


function make_update!(opt::RMSprop, path, x, gx)
    # resize accumulated and squared updates
    E_g_sq_t = get(opt.E_g_sq_t, path, zero(gx))
    # accumulate gradient
    E_g_sq_t = opt.gamma * E_g_sq_t + (1f0 - opt.gamma) * (gx .^ 2)
    # compute update
    RMS_g_t = sqrt.(E_g_sq_t .+ opt.eps)
    opt.E_g_sq_t[path] = E_g_sq_t
    return x - opt.eta .* gx ./ RMS_g_t
end


################################################################################
#                                   Adam                                       #
################################################################################


mutable struct Adam <: Optimizer
    t::Int32
    eps::Float32
    lr::Float32
    beta1::Float32
    beta2::Float32
    m_t::Dict{Any, Any}
    v_t::Dict{Any, Any}
end


function Adam(;lr::Real=0.001, beta1::Real=0.9, beta2::Real=0.999, eps::Real=10e-8)
    @assert lr > 0.0 "lr must be greater than 0"
    @assert beta1 > 0.0 "beta1 must be greater than 0"
    @assert beta2 > 0.0 "beta2 must be greater than 0"
    @assert eps > 0.0 "eps must be greater than 0"
    Adam(1, eps, lr, beta1, beta2, Dict(), Dict())
end


function make_update!(opt::Adam, path, x, gx)
    # no update for symbolic Zero
    gx isa ChainRulesCore.Zero && return x
    # resize biased moment estimates if first iteration
    m_t = get(opt.m_t, path, zero(gx))
    v_t = get(opt.v_t, path, zero(gx))
    # update biased first moment estimate
    m_t = opt.beta1 * m_t + (1f0 - opt.beta1) * gx
    # update biased second raw moment estimate
    v_t = opt.beta2 * v_t + (1f0 - opt.beta2) * (gx .^ 2)
    # save moments for the next iteration
    opt.m_t[path] = m_t
    opt.v_t[path] = v_t
    # compute bias corrected first moment estimate
    m̂_t = m_t ./ (1f0 - opt.beta1^opt.t)
    # compute bias corrected second raw moment estimate
    v̂_t = v_t ./ (1f0 - opt.beta2^opt.t)
    # apply update
    x_t1 = x - opt.lr * m̂_t ./ (sqrt.(v̂_t) .+ opt.eps)
    return x_t1
end
