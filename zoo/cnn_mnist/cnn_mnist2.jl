# Translated from PyTorch version, see cnn_minst.py

using Avalon
import Avalon: accuracy
using MLDatasets

# include("../src/core.jl")
# __init__()

mutable struct Net
    conv1::Conv2d
    conv2::Conv2d
    fc1::Linear
    fc2::Linear
end


Net() = Net(
    Conv2d(1, 20, 5),
    Conv2d(20, 50, 5),
    Linear(4 * 4 * 50, 500),
    Linear(500, 10)
)


function (m::Net)(x::AbstractArray)
    x = maxpool2d(relu.(m.conv1(x)), (2, 2))
    x = maxpool2d(relu.(m.conv2(x)), (2, 2))
    # prod(size(x)[1:3])
    x = reshape(x, 4*4*50, :)
    x = relu.(m.fc1(x))
    x = logsoftmax(m.fc2(x))
    return x
end


function get_mnist_data(train::Bool; device=CPU())
    X, Y = train ? MNIST.traindata() : MNIST.testdata()
    X = convert(Array{Float64}, reshape(X, 28, 28, 1, :)) |> device
    # replace class label like "0" with its position like "1"
    Y = Y .+ 1 |> device
    return X, Y
end


function main()
    # choose device: if CUDA is available on the system, GPU() will be used, otherwise - CPU()
    device = best_available_device()
    # instantiate the model
    m = Net() |> device
    # load training data
    X_trn, Y_trn = get_mnist_data(true);
    # set loss function and optimizer, then fit the model
    loss_fn = NLLLoss()
    opt = SGD(1e-2; momentum=0)
    @time fit!(m, X_trn, Y_trn, loss_fn; n_epochs=10, opt=opt, batch_size=100, device=device)

    # load test data
    X_tst, Y_tst = get_mnist_data(false)
    # convert to device
    X_tst = X_tst |> device; Y_tst = Y_tst |> device
    # predict log probabilities and calculate accuracy
    Ŷ = m(X_tst)
    @info accuracy(Y_tst, Ŷ)
end
