function getcpudata(nosamples = 60)
    script = "#!/bin/bash\n\nx=1\nwhile [ \$x -le $nosamples ]\ndo\n  timeout 1s top -b > cpuusg\n  sleep 2\n  grep \"%Cpu\" cpuusg >> cpudata\n  x=\$(( \$x + 1 ))\ndone\n"
    startpath = pwd()
    compilepath = gotocompile(true)
    cd(compilepath)
    open("tempscript.sh", "w") do f
        write(f, script)
    end 
    runscript = `bash tempscript.sh`
    run(runscript)
    sleep(1)
    rm("tempscript.sh")
    rm("cpuusg")
    cd(startpath)
end

function processcpudata(nosamples = 60)
    startpath = pwd()
    compath = gotocompile(true)
    cd(compath)
    data = open("cpudata") do f
        read(f,String)
    end
    rm("cpudata")
    data = split(data, "\n")
    cpudata = Array{String}[]
    for m in data
        push!(cpudata, split(m," "))
    end
    cpuusg = Float64[]
    for d in cpudata[1:nosamples]
        push!(cpuusg, parse(Float64,d[2]) + parse(Float64,d[5]))
    end
    cd(startpath)
    return cpuusg
end


"""
Returns a estimation of the GFLOPs that were performed during the functions runtime
"""
function cpugflops(nosamples = 60)
    startpath = pwd()
    compath = gotocompile(true)
    cd(compath)
    meanflops = open("src/neflops/cpuflops") do f
        read(f, String)
    end
    getcpudata(nosamples)
    meanflops = parse(Float64,meanflops)
    cpuusg = processcpudata(nosamples)
    cd(startpath)
    return meanflops * (mean(cpuusg)/100.0)
end
