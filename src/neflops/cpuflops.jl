function runcpubenchmark(x32=false,x64=true)
    startpath = pwd() 
    if ((x32 || x64) && !(x32 && x64)) && x32
        benchpath = pwd() * "/src/neflops/benchmarks/IntelCPU/linpack/"
        cd(benchpath)
        getflopestimate = `sh runme_xeon32` 
        run(getflopestimate);
        flopstring = open("lin_xeon32.txt") do f
            read(f,String)
        end
        rm("lin_xeon32.txt")
        resultstring = split(flopstring, "\n")
        startidx = Int8[] 
        endidx = Int8[]
        index = for (i,s) in enumerate(resultstring)
            if contains(s,"GFlops")
                push!(startidx,i)
            end
            if contains(s,"Done")
                push!(endidx,i)
            end
        end
        resultstring = resultstring[startidx[1]+1:endidx[1]-1]
        resultsmatrix = Array{Any}[]
        for str in resultstring
            push!(resultsmatrix, split(str," "))
        end
        for res in resultsmatrix
            filter!(x->x≠"",res)
        end
        avgflops = Float64[] 
        for num in resultsmatrix
            push!(avgflops,parse(Float64, num[5]))
        end
        approxflops = mean(avgflops)
        savepath = startpath * "/src/neflops/"
        cd(savepath)
        open("cpuflops", "w") do f
            write(f, string(approxflops))
        end
    elseif ((x64 || x32) && !(x64 && x32)) && x64
        benchpath = pwd() * "/src/neflops/benchmarks/IntelCPU/linpack/"
        cd(benchpath)
        getflopestimate = `sh runme_xeon64` 
        run(getflopestimate);
        flopstring = open("lin_xeon64.txt") do f
            read(f,String)
        end
        rm("lin_xeon64.txt")
        resultstring = split(flopstring, "\n")
        startidx = Int8[] 
        endidx = Int8[]
        index = for (i,s) in enumerate(resultstring)
            if contains(s,"GFlops")
                push!(startidx,i)
            end
            if contains(s,"Done")
                push!(endidx,i)
            end
        end
        resultstring = resultstring[startidx[1]+1:endidx[1]-1]
        resultsmatrix = Array{Any}[]
        for str in resultstring
            push!(resultsmatrix, split(str," "))
        end
        for res in resultsmatrix
            filter!(x->x≠"",res)
        end
        avgflops = Float64[] 
        for num in resultsmatrix
            push!(avgflops,parse(Float64, num[5]))
        end
        approxflops = mean(avgflops)
        savepath = startpath * "/src/neflops/"
        cd(savepath)
        open("cpuflops", "w") do f
            write(f, string(approxflops))
        end
        cd(startpath)
    else
        error("Not an allowed combination")
    end

end

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
    #rm("./tempscript.sh")
    #rm("./cpuusg")
end

function processcpudata(nosamples = 60)
    data = open("./cpudata") do f
        read(f,String)
    end
    #rm("./cpudata")
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
