module Avalon

export
    grad,
    # initialization
    init_constant!,
    init_kaiming_normal!,
    # conv
    conv1d,
    conv2d,
    conv3d,
    maxpool2d,
    # activations
    logistic,
    sigmoid,
    softplus,
    softsign,
    logsigmoid,
    relu,
    leakyrelu,
    # elu,
    softmax,
    logsoftmax,
    # losses
    nllloss,
    crossentropyloss,
    mseloss,
    NLLLoss,
    CrossEntropyLoss,
    MSELoss,
    # general layers
    Linear,
    Sequential,
    # conv layers
    Conv1d,
    Conv2d,
    Conv3d,
    # batchnorm
    BatchNorm2d,
    # RNN layers
    RNN,
    LSTM,
    GRU,
    init_hidden,
    # optim
    update!,
    SGD,
    RMSprop,
    Adam,
    # training
    fit!,
    trainmode!,
    testmode!,
    # device API (reexport from Yota)
    best_available_device
    # metrics (not exported by default)



include("core.jl")

end # module
