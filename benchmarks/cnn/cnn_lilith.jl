using Avalon
import Avalon: batchiter
using MLDatasets
using BenchmarkTools


mutable struct Net
    conv1::Conv2d
    conv2::Conv2d
    conv3::Conv2d
    fc::Linear
end


Net() = Net(
    Conv2d(1, 16, 3; padding=(1,1)),
    Conv2d(16, 32, 3; padding=(1,1)),
    Conv2d(32, 32, 3; padding=(1,1)),
    Linear(288, 10)
)

function (m::Net)(x::AbstractArray)
    x = maxpool2d(relu.(m.conv1(x)), (2, 2))
    x = maxpool2d(relu.(m.conv2(x)), (2, 2))
    x = maxpool2d(relu.(m.conv3(x)), (2, 2))
    x = reshape(x, 288, :)
    x = relu.(m.fc(x))
    x = softmax(x)
    return x
end


function get_mnist_data(train::Bool; device=CPU())
    X, Y = train ? MNIST.traindata() : MNIST.testdata()
    X = convert(Array{Float64}, reshape(X, 28, 28, 1, :)) |> device
    # replace class label like "0" with its position like "1"
    Y = Y .+ 1
    # transform to one-hot encoding
    Y1h = zeros(10, length(Y))
    for (j, y) in enumerate(Y)
        Y1h[y, j] = 1.0
    end
    Y1h = Y1h |> device
    return X, Y1h
end


function main(device=best_available_device())
    # instantiate the model
    m = Net() |> device
    # load training data
    X_trn, Y_trn = get_mnist_data(true);
    X_trn = device(X_trn)
    # set loss function and optimizer, then fit the model
    loss_fn = CrossEntropyLoss()
    opt = Adam(lr=1e-3)
    @time fit!(m, X_trn, Y_trn, loss_fn; n_epochs=10, opt=opt, batch_size=100, device=device)

    x = X_trn[:,:,:,1:100] |> device
    @benchmark m(x)
end


# function main2()
#     x = X_trn[:,:,:,1:100] |> device
#     y = Y_trn[:, 1:100] |> device

#     _, g = grad((m, x, y) -> loss_fn(m(x), y), m, x, y)
# end

# using Yota
# import Avalon: batchiter



# function _crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat)
#   return -sum(y .* log.(ŷ)) * 1 // size(y, 2)
# end


function Avalon.fit!(m::Net, X::AbstractArray, Y::AbstractArray, loss_fn;
              n_epochs=10, batch_size=100, opt=SGD(1e-3), device=CPU(), report_every=1)
    f = (m, x, y) -> loss_fn(m(x), y)
    num_batches = size(X)[end] // batch_size
    for epoch in 1:n_epochs
        epoch_loss = 0
        @time for (i, (x, y)) in enumerate(batchiter((X, Y), sz=batch_size))
            x = to_device(device, copy(x))
            y = to_device(device, copy(y))
            loss, g = grad(f, m, x, y)
            update!(opt, m, g[2])
            epoch_loss += loss
        end
        if epoch % report_every == 0
            println("Epoch $epoch: avg_loss=$(epoch_loss / num_batches)")
        end
    end
    return m
end


# using Printf
# using Yota: Tape, exec!, play!


# function profile_tape!(tape::Tape, args...; use_compiled=true, debug=false)
#     for (i, val) in enumerate(args)
#         @assert(tape[i] isa Input, "More arguments than the original function had")
#         tape[i].val = val
#     end
#     timings = Vector{Float64}(undef, 0)
#     if use_compiled && tape.compiled != nothing
#         Base.invokelatest(tape.compiled)
#     else
#         for op in tape
#             if debug
#                 println(op)
#             end
#             t = @elapsed exec!(tape, op)
#             push!(timings, t)
#         end
#     end
#     for (t, op) in zip(timings, tape)
#         @printf "%.3f\t%s\n" (t * 10^6) op
#     end
#     return tape[tape.resultid].val
# end
