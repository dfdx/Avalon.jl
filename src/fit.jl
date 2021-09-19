
"""
getbatch(dataset, i::Int, sz::Int)

Get batch of data from the specified dataset.

 * i - batch index
 * sz - size of the batch

Typical implementation for continuous datasets:

    start = (i-1)*sz + 1
    finish = min(i*sz, length(dataset))
    return start <= finish ? dataset[start:finish] : nothing
"""
function getbatch(a::AbstractArray{T,N}, i::Int, sz::Int; batch_dim=ndims(a)) where {T,N}
    start = (i-1)*sz + 1
    finish = min(i*sz, size(a, batch_dim))
    start > finish && return nothing
    selector = [d == batch_dim ? (start:finish) : (:) for d=1:ndims(a)]
    return @view a[selector...]
end


function getbatch(t::Tuple, i::Int, sz::Int)
    return ((getbatch(a, i, sz) for a in t)...,)
end


mutable struct BatchIter
    data
    sz::Int
end

Base.show(io::IO, it::BatchIter) = print(io, "BatchIter($(typeof(it.data)), sz=$(it.sz))")
batchiter(data; sz::Int=1) = BatchIter(data, sz)


function Base.iterate(it::BatchIter, i::Int=1)
    r = getbatch(it.data, i, it.sz)
    if r == nothing || (r isa Tuple && all(x -> x == nothing, r))
        return nothing
    else
        return r, i+1
    end
end


# TODO: implement `getbatch(data::ImageFolder, i, sz)` with threads


## Supervised learning with input X and output Y

# function partial_fit!(m, X::AbstractArray, Y::AbstractArray, full_loss_fn;
#                       opt=SGD(1e-3), batch_size=100, device=CPU())
#     epoch_loss = 0
#     for (i, (x, y)) in enumerate(batchiter((X, Y), size=batch_size))
#         x = to_device(device, copy(x))
#         y = to_device(device, copy(y))
#         loss, g = grad(full_loss_fn, m, x, y)
#         update!(opt, m, g[1])
#     end
#     return epoch_loss / size(X)[end]
# end


# function fit!(m, X::AbstractArray, Y::AbstractArray, loss_fn;
#               n_epochs=10, batch_size=100, opt=SGD(1e-3), device=CPU(), report_every=1)
#     full_loss_fn = (m, x, y) -> loss_fn(m(x), y)
#     for epoch in 1:n_epochs
#         time = @elapsed epoch_loss =
#             partial_fit!(m, X, Y, full_loss_fn, batch_size=batch_size, device=device)
#         if epoch % report_every == 0
#             println("Epoch $epoch: avg_cost=$(epoch_loss / (size(X,2) / batch_size)), elapsed=$time")
#         end
#     end
#     return m
# end

function fit!(m, X::AbstractArray, Y::AbstractArray, loss_fn;
              n_epochs=10, batch_size=100, opt=SGD(1e-3), device=CPU(), report_every=1)
    f = (m, x, y) -> loss_fn(m(x), y)
    num_batches = size(X)[end] // batch_size
    for epoch in 1:n_epochs
        epoch_loss = 0
        for (i, (x, y)) in enumerate(batchiter((X, Y), sz=batch_size))
            x = device(copy(x))
            y = device(copy(y))
            loss, g = grad(f, m, x, y)
            update!(opt, m, g[1])
            epoch_loss += loss
        end
        if epoch % report_every == 0
            println("Epoch $epoch: avg_loss=$(epoch_loss / num_batches)")
        end
    end
    return m
end


## Unsupervised learning with only X

function partial_fit!(m, X::AbstractArray, loss_fn;
                      opt=SGD(1e-3), batch_size=100, device=CPU())
    epoch_loss = 0
    f = (m, x) -> loss_fn(m(x))
    for (i, x) in enumerate(batchiter(X, sz=batch_size))
        x = device(copy(x))
        loss, g = grad(f, m, x)
        update!(opt, m, g[1])
        epoch_loss += loss
        println("iter $i: loss=$loss")
    end
    return epoch_loss
end


function fit!(m, X::AbstractArray, loss_fn;
              n_epochs=10, batch_size=100, opt=SGD(1e-3), device=CPU(), report_every=1)
    for epoch in 1:n_epochs
        time = @elapsed epoch_loss = partial_fit!(m, X, loss_fn, batch_size=batch_size, device=device)
        if epoch % report_every == 0
            @info("Epoch $epoch: avg_cost=$(epoch_loss / (size(X,2) / batch_size)), elapsed=$time")
        end
    end
    return m
end
