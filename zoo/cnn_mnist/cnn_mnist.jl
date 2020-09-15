## Based on PyTorch tutorial: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
## Note that that tutorial and this file use dummy inputs just to demonstrate API,
## for fully working example see cnn_mnist2.jl

using Avalon

# include("../src/core.jl")
# __init__()

mutable struct Net
    conv1::Conv2d
    conv2::Conv2d
    fc1::Linear
    fc2::Linear
    fc3::Linear    
end


Net() = Net(
    Conv2d(1, 6, 3),
    Conv2d(6, 16, 3),
    Linear(16 * 6 * 6, 120),
    Linear(120, 84),
    Linear(84, 10)
)


function (m::Net)(x::AbstractArray)
    x = maxpool2d(relu.(m.conv1(x)), (2, 2))
    x = maxpool2d(relu.(m.conv2(x)), (2, 2))
    # prod(size(x)[1:3])
    x = reshape(x, 576, :)
    x = relu.(m.fc1(x))
    x = relu.(m.fc2(x))
    x = m.fc3(x)
    return x
end


function main()
    m = Net()
    X = rand(32, 32, 1, 1000);   
    Y = rand(10, 1000);
    loss_fn = MSELoss()   
    fit!(m, X, Y, loss_fn; lr=1e-3)
end

