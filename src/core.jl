using Yota
using Base.Iterators
using Statistics
using MLDataUtils
using Distributions
import NNlib
using CUDA


include("utils.jl")
include("init.jl")
include("conv.jl")
include("rnn.jl")
include("activations.jl")
include("losses.jl")
include("layers.jl")
include("batchnorm.jl")
include("optim.jl")
include("device.jl")
include("fit.jl")
include("metrics.jl")


if CUDA.functional()
    try
        include("cuda.jl")
    catch ex
        @warn "CUDA is installed, but not working properly" exception=(ex,catch_backtrace())
    end
end


function __init__()
    register_conv_derivs()
    register_batchnorm_derivs()
    register_activation_derivs()
    register_loss_derivs()
end
