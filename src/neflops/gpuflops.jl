function getgpuflops()
    meanflops = open("gpuflops") do f
        read(f,String)
    end
    return parse(Float64,meanflops)
end