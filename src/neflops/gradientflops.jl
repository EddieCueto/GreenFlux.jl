"""
    gradientflops(layer, input)::Float64

Calculates the number of non-embedding Floating Point Operations `neFLOPs` for most layers.
`layer` should be any of the [Flux](https://github.com/FluxML/Flux.jl) model layers except
`GlobalMaxPool` and `GlobalMeanPool`.

# Example
```julia
layer = Conv((2, 2), 1=>16, relu)
input = (4,4)
layerflops(Conv((2, 2), 1=>16, relu),(4,4))
```
"""
function gradientflops(layer::Dense, input::Tuple)
    x = 0; y = 0; Fm = 1
    if length(input) == 3
        x,y,Fn = input
        xo, yo = outdims(layer,(x,y))
    elseif length(input) == 2
        x,y,_ = input
        xo, yo = outdims(layer,(x,y))
    else
        error("Not a valid Input size, expected (::Int,::Int) or (::Int,::Int,::Int)")
    end
    bi = length(layer.b)
    noofoper = gradientoperations(layer)
    return convert(Float64, ((xo*yo+bi)*noofoper)*Fm), (xo,yo)
end

function gradientflops(layer::Maxout,input::Tuple)
    x = 0; y = 0; Fm = 1
    if length(input) == 3
        x,y,Fm = input
        xo, yo = outdims(layer,(x,y))
    elseif length(input) == 2
        x,y,_ = input
        xo, yo = outdims(layer,(x,y))
    else
        error("Not a valid Input size, expected (::Int,::Int) or (::Int,::Int,::Int)")
    end
    a,b = layer.over
    return convert(Float64,xo*yo*Fm), (x,1)
end


function gradientflops(layer::Conv, input::Tuple)
    _,_,_,outputfeaturemaps = size(layer.weight)
    x = 0; y = 0; Fm = 1
    if length(input) == 3
        x,y,Fm = input
        xo, yo = outdims(layer,(x,y))
    elseif length(input) == 2
        x,y,_ = input
        xo, yo = outdims(layer,(x,y))
    else
        error("Not a valid Input size, expected (::Int,::Int) or (::Int,::Int,::Int)")
    end
    noofoper = gradientoperations(layer)
    if size(layer.bias) == ()
        return convert(Float64,xo*yo*noofoper*Fm), (xo,yo,outputfeaturemaps)
    else
        return convert(Float64,((xo*yo+size(layer.bias))*noofoper)*Fm), (xo,yo,outputfeaturemaps)
    end
end

function gradientflops(layer::MaxPool, input::Tuple)
    kernelsizex,kernelsizey = layer.k
    x = 0; y = 0; Fm = 1
    if length(input) == 3
        x,y,Fm = input
        xo, yo = outdims(layer,(x,y))
    elseif length(input) == 2
        x,y,_ = input
        xo, yo = outdims(layer,(x,y))
    else
        error("Not a valid Input size, expected (::Int,::Int) or (::Int,::Int,::Int)")
    end
    outx, outy = outdims(layer,(x,y))
    return convert(Float64,outx*outy*Fm), (outx,outy)
end

gradientflops(layer::GlobalMaxPool, input::Tuple) = error("Must be implemented in a future release")

function gradientflops(layer::MeanPool, input::Tuple)
    kernelsizex,kernelsizey = layer.k
    x = 0; y = 0; Fm = 1
    if length(input) == 3
        x,y,Fm = input
        xo, yo = outdims(layer,(x,y))
    elseif length(input) == 2
        x,y,_ = input
        xo, yo = outdims(layer,(x,y))
    else
        error("Not a valid Input size, expected (::Int,::Int) or (::Int,::Int,::Int)")
    end
    outx, outy = outdims(layer,(x,y))
    return convert(Float64,outx*outy*Fm), (outx,outy)
end

gradientflops(layer::GlobalMeanPool, input::Tuple) = error("Must be implemented in a future release")

function gradientflops(layer::DepthwiseConv, input::Tuple)
    kernelsizex,kernelsizey,inputfeaturemaps,outputfeaturemaps = size(layer.weight)
    x = 0; y = 0; Fm = 1
    if length(input) == 3
        x,y,Fm = input
        xo, yo = outdims(layer,(x,y))
    elseif length(input) == 2
        x,y,_ = input
        xo, yo = outdims(layer,(x,y))
    else
        error("Not a valid Input size, expected (::Int,::Int) or (::Int,::Int,::Int)")
    end
    outx, outy = outdims(layer,(x,y))
    noofoper = gradientoperations(layer)
    if size(layer.bias) == ()
        return convert(Float64,kernelsizex*kernelsizey*noofoper*Fm), (outx,outy)
    else
        return convert(Float64,kernelsizex*kernelsizey*noofoper*Fm + (length(layer.bias)-1)), (outx,outy)
    end
end

function gradientflops(layer::ConvTranspose, input::Tuple)
    kernelsizex,kernelsizey,inputfeaturemaps,outputfeaturemaps = size(layer.weight)
    x = 0; y = 0; Fm = 1
    if length(input) == 3
        x,y,Fm = input
        xo, yo = outdims(layer,(x,y))
    elseif length(input) == 2
        x,y,_ = input
        xo, yo = outdims(layer,(x,y))
    else
        error("Not a valid Input size, expected (::Int,::Int) or (::Int,::Int,::Int)")
    end
    outx, outy = outdims(layer,(x,y))
    noofoper = gradientoperations(layer)
    if size(layer.bias) == ()
        return convert(Float64,kernelsizex*kernelsizey*noofoper*Fm), (outx,outy)
    else
        return convert(Float64,kernelsizex*kernelsizey*noofoper*Fm + (length(layer.bias)-1)), (outx,outy)
    end
end

function gradientflops(layer::CrossCor, input::Tuple)
    kernelsizex,kernelsizey,inputfeaturemaps,outputfeaturemaps = size(layer.weight)
    x = 0; y = 0; Fm = 1
    if length(input) == 3
        x,y,Fm = input
        xo, yo = outdims(layer,(x,y))
    elseif length(input) == 2
        x,y,_ = input
        xo, yo = outdims(layer,(x,y))
    else
        error("Not a valid Input size, expected (::Int,::Int) or (::Int,::Int,::Int)")
    end
    outx, outy = outdims(layer,(x,y))
    noofoper = gradientoperations(layer)
    if size(layer.bias) == ()
        return convert(Float64,kernelsizex*kernelsizey*noofoper*Fm), (outx,outy)
    else
        return convert(Float64,kernelsizex*kernelsizey*noofoper*Fm + (length(layer.bias)-1)), (outx,outy)
    end
end

function gradientflops(layer::Recur{T}, input::Tuple) where {T <: RNNCell}
    inM,inN = input
    WhM,WhN = size(layer.cell.Wh)
    WiM,WiN = size(layer.cell.Wi)
    hM = length(layer.cell.h)
    noofoper = activationoperations(layer)
    if size(layer.cell.b) == ()
        return convert(Float64,((2*WhN*WhM-hM + 2*WiM*WiN-inM)*noofoper)+((2*WhN*WhM-hM)*noofoper)), (hM,1)
    else
        bM = length(layer.cell.b)
        return convert(Float64,((2*WhN*WhM-hM + 2*WiM*WiN-inM + 2*bM)*noofoper)+((2*WhN*WhM-hM + bM)*noofoper)), (bM,1)
    end
end

function gradientflops(layer::Recur{T}, input::Tuple) where {T <: LSTMCell}
    inM,inN = input
    WhM,WhN = size(layer.cell.Wh)
    WiM,WiN = size(layer.cell.Wi)
    hM = length(layer.cell.h)
    noofoper = 3
    if size(layer.cell.b) == ()
        return convert(Float64,((2*WhN*WhM-hM + 2*WiM*WiN-inM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*WiN-inM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*WiN-inM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*WiN-inM)*noofoper)+(3*hM)+((hM*noofoper)+hM)), (hM,1)
    else
        bM = length(layer.cell.b)
        return convert(Float64,((2*WhN*WhM-hM + 2*WiM*WiN-inM + 2*bM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*WiN-inM + 2*bM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*WiN-inM + 2*bM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*WiN-inM + 2*bM)*noofoper)+(3*bM)+((bM*noofoper)+bM)), (bM,1)
    end
end

function gradientflops(layer::Recur{T}, input::Tuple) where {T <: GRUCell}
    inM,inN = input
    WhM,WhN = size(layer.cell.Wh)
    WiM,WiN = size(layer.cell.Wi)
    hM = length(layer.cell.h)
    noofoper = 3
    if size(layer.cell.b) == ()
        return convert(Float64,((2*WhN*WhM-hM + 2*WiM*WiN-inM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*WiN-inM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*WiN-inM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*WiN-inM)*noofoper)+(3*hM)+((hM*noofoper)+hM)), (hM,1)
    else
        bM = length(layer.cell.b)
        return convert(Float64,((2*WhN*WhM-hM + 2*WiM*WiN-inM + 2*bM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*WiN-inM + 2*bM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*(WiN+hM)-inM + 2*bM)*noofoper)+(4bM)), (bM,1)
    end
end
