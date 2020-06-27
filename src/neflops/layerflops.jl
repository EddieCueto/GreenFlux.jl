"""
    layerflops(layer, input)::Float64

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
function layerflops(layer::Dense,input::Tuple)
    N,M = size(layer.W)
    Mi = 0; Ni = 0; Fm = 1; out = 0
    if length(input) == 3
        Mi,Ni,Fm = input
        out = outdims(layer,(Mi,Ni))
    elseif length(input) == 2
        Mi,Ni = input
        Fm = 1
        out = outdims(layer,input)
    else
        error("Not a valid Input size, expected (::Int,::Int) or (::Int,::Int,::Int)")
    end
    bi = length(layer.b)
    if length(out) < 2
        out = (out[1],1)
    end
    noofopers = activationoperations(layer)
    return convert(Float64,((2*Mi*N - M)+bi)*noofopers*Fm), out
end

function layerflops(layer::Maxout,input::Tuple)
    i = 0; j = 0; Fm = 1
    if length(input) == 3
        i,j,Fm = input
    elseif length(input) == 2
        i,j = input
        Fm = 1
    else
        error("Not a valid Input size, expected (::Int,::Int) or (::Int,::Int,::Int)")
    end
    a,b = layer.over
    return convert(Float64,a*b*j*i*Fm), (j,1)
end


function layerflops(layer::Conv, input::Tuple)
    kernelsizex,kernelsizey,inputfeaturemaps,outputfeaturemaps = size(layer.weight)
    x = 0; y = 0
    if length(input) == 3
        x, y, _ = input
    elseif length(input) == 2
        x, y = input
    else
        error("Not a valid Input size, expected (::Int,::Int) or (::Int,::Int,::Int)")
    end
    outx, outy = outdims(layer,(x,y))
    noofoper = activationoperations(layer)
    if size(layer.bias) == ()
        return convert(Float64,inputfeaturemaps*(kernelsizex*kernelsizey+(kernelsizex*kernelsizey-1))*outx*outy + outx*outy*noofoper*outputfeaturemaps), (outx,outy,outputfeaturemaps)
    else
        return convert(Float64,(2*kernelsizex*kernelsizey)*inputfeaturemaps*outx*outy + outx*outy*noofoper*outputfeaturemaps), (outx,outy,outputfeaturemaps)
    end
end

function layerflops(layer::MaxPool, input::Tuple)
    Fm = 1
    if length(input) == 3
        _,_,Fm = input
    elseif length(input) == 2
        _,_ = input
        Fm = 1
    else
        error("Not a valid Input size, expected (::Int,::Int) or (::Int,::Int,::Int)")
    end
    kernelsizex,kernelsizey = layer.k
    outx, outy = outdims(layer,input)
    return convert(Float64,kernelsizex*kernelsizey*outx*outy*Fm), (outx,outy)
end

layerflops(layer::GlobalMaxPool, input::Tuple) = error("Must be implemented in a future release")

function layerflops(layer::MeanPool, input::Tuple)
    Fm = 1
    if length(input) == 3
        _,_,Fm = input
    elseif length(input) == 2
        _,_ = input
        Fm = 1
    else
        error("Not a valid Input size, expected (::Int,::Int) or (::Int,::Int,::Int)")
    end
    kernelsizex,kernelsizey = layer.k
    outx, outy = outdims(layer,input)
    return convert(Float64,kernelsizex*kernelsizey*outx*outy*Fm), (outx,outy)
end

layerflops(layer::GlobalMeanPool, input::Tuple) = error("Must be implemented in a future release")

function layerflops(layer::DepthwiseConv, input::Tuple)
    kernelsizex,kernelsizey,inputfeaturemaps,outputfeaturemaps = size(layer.weight)
    x = 0; y = 0
    if length(input) == 3
        x, y, _ = input
    elseif length(input) == 2
        x, y = input
    else
        error("Not a valid Input size, expected (::Int,::Int) or (::Int,::Int,::Int)")
    end
    outx, outy = outdims(layer,(x,y))
    noofoper = activationoperations(layer)
    if size(layer.bias) == ()
        return convert(Float64,inputfeaturemaps*(kernelsizex*kernelsizey+(kernelsizex*kernelsizey-1))*outx*outy + outx*outy*noofoper*outputfeaturemaps), (outx,outy,outputfeaturemaps)
    else
        return convert(Float64,(2*kernelsizex*kernelsizey)*inputfeaturemaps*outx*outy + outx*outy*noofoper*outputfeaturemaps), (outx,outy,outputfeaturemaps)
    end
end

function layerflops(layer::ConvTranspose, input::Tuple)
    kernelsizex,kernelsizey,inputfeaturemaps,outputfeaturemaps = size(layer.weight)
    x = 0; y = 0
    if length(input) == 3
        x, y, _ = input
    elseif length(input) == 2
        x, y = input
    else
        error("Not a valid Input size, expected (::Int,::Int) or (::Int,::Int,::Int)")
    end
    outx, outy = outdims(layer,(x,y))
    noofoper = activationoperations(layer)
    if size(layer.bias) == ()
        return convert(Float64,inputfeaturemaps*(kernelsizex*kernelsizey+(kernelsizex*kernelsizey-1))*outx*outy + outx*outy*noofoper*outputfeaturemaps), (outx,outy,outputfeaturemaps)
    else
        return convert(Float64,(2*kernelsizex*kernelsizey)*inputfeaturemaps*outx*outy + outx*outy*noofoper*outputfeaturemaps), (outx,outy,outputfeaturemaps)
    end
end

function layerflops(layer::CrossCor, input::Tuple)
    kernelsizex,kernelsizey,inputfeaturemaps,outputfeaturemaps = size(layer.weight)
    x = 0; y = 0
    if length(input) == 3
        x, y, _ = input
    elseif length(input) == 2
        x, y = input
    else
        error("Not a valid Input size, expected (::Int,::Int) or (::Int,::Int,::Int)")
    end
    outx, outy = outdims(layer,(x,y))
    noofoper = activationoperations(layer)
    if size(layer.bias) == ()
        return convert(Float64,inputfeaturemaps*(kernelsizex*kernelsizey+(kernelsizex*kernelsizey-1))*outx*outy + outx*outy*noofoper*outputfeaturemaps), (outx,outy,outputfeaturemaps)
    else
        return convert(Float64,(2*kernelsizex*kernelsizey)*inputfeaturemaps*outx*outy + outx*outy*noofoper*outputfeaturemaps), (outx,outy,outputfeaturemaps)
    end
end

function layerflops(layer::Recur{T}, input::Tuple) where {T <: RNNCell}
    inM = 0; inN = 0; Fm = 1
    if length(input) == 3
        inM,inN, Fm = input
    elseif length(input) == 2
        inM,inN = input
    else
        error("Not a valid Input size, expected (::Int,::Int) or (::Int,::Int,::Int)")
    end
    WhM,WhN = size(layer.cell.Wh)
    WiM,WiN = size(layer.cell.Wi)
    hM = length(layer.cell.h)
    noofoper = activationoperations(layer)
    if size(layer.cell.b) == ()
        return convert(Float64,(((2*WhN*WhM-hM + 2*WiM*WiN-inM)*noofoper)+((2*WhN*WhM-hM)*noofoper))*Fm), (hM,1)
    else
        bM = length(layer.cell.b)
        return convert(Float64,(((2*WhN*WhM-hM + 2*WiM*WiN-inM + 2*bM)*noofoper)+((2*WhN*WhM-hM + bM)*noofoper))*Fm), (bM,1)
    end
end

function layerflops(layer::Recur{T}, input::Tuple) where {T <: LSTMCell}
    inM = 0; inN = 0; Fm = 1
    if length(input) == 3
        inM,inN, Fm = input
    elseif length(input) == 2
        inM,inN = input
    else
        error("Not a valid Input size, expected (::Int,::Int) or (::Int,::Int,::Int)")
    end
    WhM,WhN = size(layer.cell.Wh)
    WiM,WiN = size(layer.cell.Wi)
    hM = length(layer.cell.h)
    noofoper = 3
    if size(layer.cell.b) == ()
        return convert(Float64,(((2*WhN*WhM-hM + 2*WiM*WiN-inM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*WiN-inM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*WiN-inM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*WiN-inM)*noofoper)+(3*hM)+((hM*noofoper)+hM))*Fm), (hM,1)
    else
        bM = length(layer.cell.b)
        return convert(Float64,(((2*WhN*WhM-hM + 2*WiM*WiN-inM + 2*bM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*WiN-inM + 2*bM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*WiN-inM + 2*bM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*WiN-inM + 2*bM)*noofoper)+(3*bM)+((bM*noofoper)+bM))*Fm), (bM,1)
    end
end

function layerflops(layer::Recur{T}, input::Tuple) where {T <: GRUCell}
    inM = 0; inN = 0; Fm = 1
    if length(input) == 3
        inM,inN, Fm = input
    elseif length(input) == 2
        inM,inN = input
    else
        error("Not a valid Input size, expected (::Int,::Int) or (::Int,::Int,::Int)")
    end
    WhM,WhN = size(layer.cell.Wh)
    WiM,WiN = size(layer.cell.Wi)
    hM = length(layer.cell.h)
    noofoper = 3
    if size(layer.cell.b) == ()
        return convert(Float64,(((2*WhN*WhM-hM + 2*WiM*WiN-inM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*WiN-inM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*WiN-inM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*WiN-inM)*noofoper)+(3*hM)+((hM*noofoper)+hM))*Fm), (hM,1)
    else
        bM = length(layer.cell.b)
        return convert(Float64,(((2*WhN*WhM-hM + 2*WiM*WiN-inM + 2*bM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*WiN-inM + 2*bM)*noofoper)+((2*WhN*WhM-hM + 2*WiM*(WiN+hM)-inM + 2*bM)*noofoper)+(4bM))*Fm), (bM,1)
    end
end
