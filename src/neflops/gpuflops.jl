function startingbenchmark()
    startpath = pwd()
    compilepath = gotocompile(true)
    cd("src/neflops/Nvidia")
    command = `nsys nvprof julia gpubenchmark.jl`
    
end

function getgpuflops()
    meanflops = open("gpuflops") do f
        read(f,String)
    end
    return parse(Float64,meanflops)
end