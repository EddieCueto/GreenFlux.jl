module GreenFlux

using Flux: Chain, Recur, Dense, Conv, MaxPool, GlobalMaxPool, MeanPool, GlobalMeanPool,
            DepthwiseConv, ConvTranspose, CrossCor, GRUCell, LSTMCell,RNNCell, Maxout
using Flux: params, outdims, sigmoid, rrelu, elu, celu, softsign, softplus, tanh, gelu, 
            hardsigmoid, logsigmoid,swish, selu, softmax, logsoftmax, hardtanh,
            leakyrelu, relu6, lisht, tanhshrink, logcosh, mish, relu, trelu,
            softshrink, identity
#using Dates
using Statistics: mean
using CUDA
using CUDA: has_cuda_gpu 

export avgpowerdraw, modelflops, cpuflops

include("power/powerdraw.jl")
include("neflops/measureutils.jl")
include("neflops/layerflops.jl")
include("neflops/gradientflops.jl")
include("neflops/modelflops.jl")
include("neflops/benchmark.jl")
include("neflops/cpuflops.jl")
include("utils.jl")

function __init__()
   @info "Finished loading GreenFlux..." 
end

end # module
