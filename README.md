# Avalon

![Status](https://github.com/dfdx/Avalon.jl/actions/workflows/test.yml/badge.svg?branch=main)


**Avalon** is a deep learning library in Julia with focus on **high performance** and **interoperability with existing DL frameworks**. Its main features include:

* tracing autograd engine - models are just structs, transformations are just functions
* optimizing code generator based on hackable computational graph
* GPU support
* layer API similar to PyTorch's to ease translation of existing Python code to Julia
* high backward compatibility to allow accumulation of models

## Usage

To get you a feeling of what Avalon is like, here's a definition of a small convolutional neural network:

```julia
using Avalon


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
    x = reshape(x, 4*4*50, :)
    x = relu.(m.fc1(x))
    x = logsoftmax(m.fc2(x))
    return x
end
```

For detailed explanation of this and other models see [the tutorial](https://github.com/dfdx/Avalon.jl/tree/master/tutorial). Some predefined models are also available in [the zoo](https://github.com/dfdx/Avalon.jl/tree/master/zoo).


## Performance

Performance comparison between different libraries is hard and benchmarks are rarely fair, but here's our best shot in this direction:

### Convolutional neural network

Code available [here](https://github.com/dfdx/Avalon.jl/tree/master/benchmarks/cnn)

|               | training 1 epoch | training total time* | prediction |
| ------------- | ---------------- | -------------------- | ---------- |
| Avalon (CPU)  |    170 s         |       1742 s         |   39 ms    |
| Flux (CPU)    |    250 s         |       2515 s         |   42 ms    |
| ------------- | ---------------- | -------------------- | ---------- |
| Avalon (GPU)  |     10 s         |        164 s         |    5 ms    |
| Flux (GPU)    |     12 s         |        150 s         |    5 ms    |
| PyTorch (GPU) |     12 s         |        120 s         |    2 ms    |

`*` - total time includes 10 epochs + compilation time

Note that in the test on GPU Avalon has longest compilation time and thus
longest total training time _after 10 epochs_. However, time per epoch
is the lowest, so Avalon is typically the fastest one in longer run.



### Variational Autoencoder

Code available [here](https://github.com/dfdx/Avalon.jl/tree/master/benchmarks/vae)

|               | training 1 epoch | training total time  | prediction |
| ------------- | ---------------- | -------------------- | ---------- |
| Avalon (CPU)  |     50 s         |        535 s         |   395 μs   |
| Flux (CPU)    |    948 s         |        158 min       |    81 ms   |
| ------------- | ---------------- | -------------------- | ---------- |
| Avalon (GPU)  |      3 s         |         93 s         |   194 μs   |
| Flux (GPU)**  |     ---          |          ---         |     ---    |
| PyTorch (GPU) |      7 s         |         66 s         |   501 µs   |

`**` - VAE example from the Flux zoo doesn't work on GPU


## API Stability

One of the central ideas behind Avalon is the ability to reuse existing code instead of writing everything from scratch.
To facilitate it, Avalon is committed to high, although not absolute backward compatibility. The following table
outlines stability level you should expect from various components of the library.

| Component       | API Stable? |
| --------------- | ----------- |
| Basic layers    | Yes         |
| CNN             | Yes         |
| RNN             | No*         |
| Losses          | Mostly      |
| Activations     | Yes         |
| Initializations | Mostly      |
| Optimizers      | Yes         |
| Device API      | Yes         |
| Fitting API     | No**        |

`*` - currently Avalon provides only basic implementations of vanilla RNN, LSTM and GRU; this implementation will be improved in future version and made more compatible with PyTorch version, but currently it cannot be considered stable

`**` - function `fit!()` provides a convenient shortcut for training supervised learning models, but in its current state it's too basic for most real use cases; for more durable code consider writing your own method for training using `fit!()` as a template

Please note that until version 1.0 "stable API" means that we will try our best to keep it unchanged, but we reserve the right to the break the rule in some rare and exceptional cases.
