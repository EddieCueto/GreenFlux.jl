function runcpubenchmark(;x32::Bool=false,x64::Bool=true)
    startpath = pwd()
    compath = gotocompile(true)
    if ((x32 || x64) && !(x32 && x64)) && x32
        benchpath = "src/neflops/benchmarks/IntelCPU/linpack/"
        getflopestimate = `sh $(benchpath)runme_xeon32` 
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
        cd(compath)
        savepath = "src/neflops/"
        open(savepath*"cpuflops", "w") do f
            write(f, string(approxflops))
        end
    elseif ((x64 || x32) && !(x64 && x32)) && x64
        benchpath = "src/neflops/benchmarks/IntelCPU/linpack/"
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
        cd(compath)
        savepath = "src/neflops/"
        open(savepath*"cpuflops", "w") do f
            write(f, string(approxflops))
        end
    else
        error("Not an allowed combination")
    end
    cd(startpath)
end