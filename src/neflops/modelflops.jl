"""
    modelflops(model::Chain)::Float64
    
Calculates the approximate number of Floating Point Operations that the model will require


```julia
weight = rand(Float64, 3, 3, 5)
bias = zeros(Float64, 5)
Conv(weight = weight,
    bias = bias,
    σ = sigmoid)
```
"""
function modelflops(model::Chain,inputsize::Tuple,samplesize::Float64,batchsize::Float64)
    x = 0; y = 0; Fm = 1
    if length(inputsize) == 3
        x,y,Fm = inputsize
    elseif length(inputsize) == 2
        x,y = inputsize
    end
    modellayers = collect(model)
    nelayers = Array{Any,1}()
    layeroutput = Array{Tuple,1}()
    outsizes = Array{Tuple,1}()
    lossandgradient = 1
    for ml in modellayers
        if islayer(ml)
            push!(nelayers,ml)
        end
    end
    output = (0,0)
    for mli in 1:length(nelayers)
        if mli == 1
            noflops, output = layerflops(nelayers[mli],inputsize)
            layeroutput = vcat(layeroutput, noflops)
        else
            noflops, output = layerflops(nelayers[mli],output)
            layeroutput = vcat(layeroutput, noflops)
        end
    end
    numberoflayers = length(layeroutput)
    return sum(layeroutput) * samplesize + lossandgradient*batchsize
end
