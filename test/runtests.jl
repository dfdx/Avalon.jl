using Avalon
import Avalon: accuracy, recall, precision, confusion_matrix
import Avalon: RNNCell, LSTMCell, GRUCell, rnn_forward, lstm_forward, gru_forward
import Avalon: ∇batchnorm2d
import Avalon.Yota: gradcheck
using Random
using Test
import CUDA
import ChainRulesTestUtils: test_rrule


Random.seed!(108);

# include("gradcheck.jl")
include("bcast.jl")
include("conv.jl")
include("rnn.jl")
# include("activations.jl")
# include("layers.jl")
# # include("losses.jl")  -- ignored since nllloss() doesn't provide derivative w.r.t. 2nd argument
# include("optim.jl")
# include("metrics.jl")

# if CUDA.functional()
#     include("cuda.jl")
# end
