# parameter initialization
# mostly based on: https://github.com/pytorch/pytorch/blob/d58059bc6fa9b5a0c9a3186631029e4578ca2bbd/torch/nn/init.py

function calculate_fan_in_and_fan_out(W::AbstractMatrix)
    return size(W, 2), size(W, 1)
end

function calculate_fan_in_and_fan_out(W::AbstractArray{T,4}) where T
    num_input_fmaps = size(W, 3)    
    num_output_fmaps = size(W, 4)
    receptive_field_size = prod(size(W)[1:2])
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out
end


function calculate_gain(nonlinearity, param=nothing)
    linear_fns = (:linear, :conv1d, :conv2d, :conv3d, :conv_transpose1d, :conv_transpose2d, :conv_transpose3d)
    if nonlinearity in linear_fns || nonlinearity == :sigmoid
        return 1
    elseif nonlinearity == :tanh
        return 5.0 / 3
    elseif nonlinearity == :relu
        return sqrt(2.0)
    elseif nonlinearity == :leaky_relu
        if param == nothing
            negative_slope = 0.01
        elseif !isa(param, Bool) && isa(param, Integer) || isa(param, Real)
            negative_slope = param
        else
            error("negative_slope $param not a valid number")
        end
        return sqrt(2.0 / (1 + negative_slope ^ 2))
    else
        error("Unsupported nonlinearity $nonlinearity")
    end
end


function init_kaiming_normal!(W::AbstractArray; mode::Symbol=:fan_in, nonlinearity=:leaky_relu, a=0)
    @assert mode in (:fan_in, :fan_out)
    fan_in, fan_out = calculate_fan_in_and_fan_out(W)
    fan = mode == :fan_in ? fan_in : fan_out
    gain = calculate_gain(nonlinearity, a)
    std = gain / sqrt(fan)
    W .= rand(Normal(0, std), size(W))
    return W
end


function init_constant!(W::AbstractArray, val)
    W .= val
end
