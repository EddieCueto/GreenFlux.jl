"""
    activationoperations(layer)::Int
    
Outputs the approximate mumber of operations for a Flux activation function.

```julia
layer = Conv(weight = weight,
    σ = sigmoid)

activationoperations(Conv(weight = weight,
    σ = sigmoid))
```
"""
function activationoperations(layer::Recur)
    layer = recurrentunpack(layer)
    activation = layer.σ
    if activation == sigmoid || activation == rrelu || activation == elu || activation == celu || 
        activation == softsign || activation == softplus || activation == tanh
        noofoper = 3 
    elseif activation == gelu
        noofoper = 8
    elseif activation == hardsigmoid || activation == logsigmoid || activation == swish ||
            activation == selu || activation == softmax || activation == logsoftmax
        noofoper = 4
    elseif activation == hardtanh || activation == leakyrelu || activation == relu6 || 
            activation == lisht || activation == tanhshrink
        noofoper = 2
    elseif activation == logcosh || activation == mish
        noofoper = 5
    elseif activation == relu || activation == trelu || activation == softshrink
        noofoper = 1
    elseif activation == identity
        noofoper = 0
    else
        @info "Unkown activation type defaulting to identity"
        return 0
    end
    return noofoper
end


function activationoperations(layer)
    activation = layer.σ
    if activation == sigmoid || activation == rrelu || activation == elu || activation == celu || 
        activation == softsign || activation == softplus || activation == tanh
        noofoper = 4 
    elseif activation == gelu
        noofoper = 9
    elseif activation == hardsigmoid || activation == logsigmoid || activation == swish ||
            activation == selu || activation == softmax || activation == logsoftmax
        noofoper = 5
    elseif activation == hardtanh || activation == leakyrelu || activation == relu6 || 
            activation == lisht || activation == tanhshrink
        noofoper = 3
    elseif activation == logcosh || activation == mish
        noofoper = 6
    elseif activation == relu || activation == trelu || activation == softshrink
        noofoper = 2
    elseif activation == identity
        noofoper = 1
    else
        @info "Unkown activation type defaulting to identity"
        return 1
    end
    return noofoper
end

"""
    gradientoperations(layer)::Int
    
Outputs the approximate mumber of operations for the gradient of a Flux activation 
function.

```julia
layer = Conv(weight = weight,
    σ = sigmoid)

gradientoperations(Conv(weight = weight,
    σ = sigmoid))
```
"""
function gradientoperations(layer::Recur)
    layer = recurrentunpack(layer)
    activation = layer.σ
    if activation == sigmoid || activation == rrelu || activation == elu || activation == celu || 
        activation == softsign || activation == softplus || activation == tanh
        noofoper = 4 
    elseif activation == gelu
        noofoper = 8
    elseif activation == hardsigmoid || activation == logsigmoid || activation == swish ||
            activation == selu || activation == softmax || activation == logsoftmax
        noofoper = 4
    elseif activation == hardtanh || activation == leakyrelu || activation == relu6 || 
            activation == lisht || activation == tanhshrink
        noofoper = 2
    elseif activation == logcosh || activation == mish
        noofoper = 5
    elseif activation == relu || activation == trelu || activation == softshrink
        noofoper = 1
    elseif activation == identity
        noofoper = 0
    else
        @info "Unkown activation type defaulting to identity"
        return 1
    end
    return noofoper
end

function gradientoperations(layer)
    activation = layer.σ
    if activation == sigmoid || activation == rrelu || activation == elu || activation == celu || 
        activation == softsign || activation == softplus || activation == tanh
        noofoper = 4 
    elseif activation == gelu
        noofoper = 8
    elseif activation == hardsigmoid || activation == logsigmoid || activation == swish ||
            activation == selu || activation == softmax || activation == logsoftmax
        noofoper = 4
    elseif activation == hardtanh || activation == leakyrelu || activation == relu6 || 
            activation == lisht || activation == tanhshrink
        noofoper = 2
    elseif activation == logcosh || activation == mish
        noofoper = 5
    elseif activation == relu || activation == trelu || activation == softshrink
        noofoper = 1
    elseif activation == identity
        noofoper = 0
    else
        @info "Unkown activation type defaulting to identity"
        return 1
    end
    return noofoper
end

function recurrentunpack(layer::Recur)
    return layer.cell
end

islayer(::Any) = false
islayer(::Recur) = true
islayer(::Dense) = true
islayer(::Conv) = true
islayer(::MaxPool) = true
islayer(::MeanPool) = true
islayer(::DepthwiseConv) = true
islayer(::ConvTranspose) = true
islayer(::CrossCor) = true
islayer(::Maxout) = true
             