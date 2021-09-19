using Yota
using Base.Iterators
using Statistics
using MLDataUtils
using Distributions
import NNlib
using CUDA
import ChainRulesCore: rrule, rrule_via_ad, NoTangent, ZeroTangent, @thunk, unthunk

include("yota_ext.jl")
include("utils.jl")
include("bcast.jl")
include("init.jl")
include("conv.jl")
include("rnn.jl")
include("activations.jl")
include("losses.jl")
include("layers.jl")
include("batchnorm.jl")
include("optim.jl")
include("fit.jl")
include("metrics.jl")
include("cuda.jl")


# if CUDA.functional()
#     try
#         include("cuda.jl")
#     catch ex
#         @warn "CUDA is installed, but not working properly" exception=(ex,catch_backtrace())
#     end
# end


# function __init__()
#     register_conv_derivs()
#     register_batchnorm_derivs()
#     register_activation_derivs()
#     register_loss_derivs()
# end
