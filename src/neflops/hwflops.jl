
function getcpuclock()
    getclkspd = `lscpu`
    cpuclockspeed = read(getclkspd, String);
    cpuclockspeed = split(cpuclockspeed, "\n")
    cpuclockspeed = split(cpuclockspeed[15], " ")
    cpuclockspeed = filter(x->x≠"",cpuclockspeed)
    cpuclockspeed = parse(Float64,cpuclockspeed[3])
    return cpuclockspeed
end


function getcpudata(nosamples = 60)
    script = "#!/bin/bash\n\nx=1\nwhile [ \$x -le $nosamples ]\ndo\n  timeout 1s top -b > cpuusg\n  sleep 2\n  grep \"%Cpu\" cpuusg >> cpudata\n  x=\$(( \$x + 1 ))\ndone\n"
    open("./tempscript.sh", "w") do f
        write(f, script)
    end 
    runscript = `bash tempscript.sh`
    run(runscript)
    sleep(1)
    rm("./tempscript.sh")
    rm("./cpuusg")
end

function processcpudata(nosamples = 60)
    data = open("./cpudata") do f
        read(f,String)
    end
    rm("./cpudata")
    data = split(data, "\n")
    cpudata = Array{String}[]
    for m in data
        push!(cpudata, split(m," "))
    end
    cpuusg = Float64[]
    for d in cpudata[1:nosamples]
        push!(cpuusg, parse(Float64,d[3]) + parse(Float64,d[6]))
    end
    return cpuusg
end


"""
Returns a estimation of the GFLOPs that were performed during the functions runtime
"""
function estcpuflops(nosamples = 60)
    meanflops = open("./src/neflops/cpuflops") do f
        read(f, String)
    end
    getcpudata(nosamples)
    meanflops = parse(Float64,meanflops)
    cpuusg = processcpudata(nosamples)
    return meanflops * (mean(cpuusg)/100.0)
end
