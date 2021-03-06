"""
    gpupowerdraw()::Float64

The function uses Linux `nvidia-smi` package to sample and get the average electricity
draw of the GPUs.
"""
function gpupowerdraw()
    gpucommand = `nvidia-smi`
    usage = Array{Any}(undef,60)
    cap = Array{Any}(undef,60)
    nogpus = 0

    for count in 1:60
        smis = Array{Any}[]
        smiss = Array{Any}[]
        gpus = Array{Any}[]
        powerdraw = Array{Float64}[]
        powercap = Array{Float64}[]
        smi = read(gpucommand, String);
        smi = split(smi, "\n")
        for s in smi
            push!(smis,split(s, " "))
        end
        for s in smis
            push!(smiss,filter(x->x≠"",s))
        end
        for strings in smiss
            if length(strings) > 5 && strings[6] == "/" && strings[10] == "/"
                push!(gpus,strings)
            end
        end
        nogpus = length(gpus)
        for g in gpus
            usagestr = ""
            capstr = ""
            if g[5] == "N/A"
                usagestr = "0W"
            else
                usagestr = usagestr * g[5]
            end
            if g[7] == "N/A"
                capstr = "0W"
            else
                capstr = capstr * g[7]
            end
            regexw = r"(\d+)"
            wattusg = match(regexw,usagestr)
            wattcap = match(regexw,capstr)
            powerdraw = vcat(powerdraw, parse(Float64,wattusg.match))
            powercap = vcat(powercap, parse(Float64,wattcap.match))
        end
        usage[count] = mean(powerdraw)
        cap[count] = mean(powercap)
        sleep(1)
    end
    return nogpus, mean(usage), mean(cap)
end


"""
    cpupowerdraw()::Float64

This function uses the Linux `powerstat` utility to get the average CPU energy cost.
"""
function cpupowerdraw()
    cpucommand = `powerstat -R -n -d0`
    cpu = read(cpucommand, String);
    cpu = split(cpu,"\n")
    cpu = cpu[66][60:64]
    return parse(Float64,cpu)
end 


#TODO: further fine tune the model
"""
    rampowerdraw()::Float64

[Approximate RAM Power Draw](https://www.jedec.org/) the values are provided by the JEDEC we just take the 
average of the total RAM with its Watt value to get the approximate power draw.  
"""
function rampowerdraw(ramtype="DDR3")
    ramcommand = `free -m`
    powerused = Array{Float64}(undef,60)
    for count in 1:60
        ram = read(ramcommand, String);
        ram = split(ram,"\n")
        ram = split(ram[2]," ")
        filter!(x->x≠"",ram)
        totalram = parse(Float64,ram[2])
        if ramtype == "DDR3"
            powerused[count] = (totalram/1024)*0.3125
        elseif ramtype == "DDR2"
            powerused[count] = (totalram/1024)*0.625
        elseif ramtype == "DDR"
            powerused[count] = (totalram/1024)*6.5
        else
            error("$ramtype unrecognized RAM type.")
        end
        sleep(1)
    end
    return mean(powerused)
end


#TODO: modify the code to work in Windows.
"""
    avgpowerdraw()::Float64

[Average Power Draw](https://arxiv.org/abs/1906.02243) where `pc` is the average power
draw (in watts) from all CPU sockets during training, let `pr` be the average power draw from all
DRAM (main memory) sockets, let `pg` be the average power draw of the GPUs during training and `g` 
the number of available gpus.

`apd = 1.58*t*(pc + pr + g*pg)/1000`

returns the average power consumption in kWh.

By default it assumes you use `"DDR3"` memory but you can pass `"DDR2"` or `"DDR"` to get a better estimate.
"""
function avgpowerdraw(freeram="DDR3")
    g, pg, pc, pr = 0.0, 0.0, 0.0, 0.0
    starttime = time()
    try
        g, pg, _ = gpupowerdraw()
    catch ex
        println(ex.msg)
        return 0.0
    end
    try
        pc = cpupowerdraw()   
    catch ex
        println(ex.msg)
        return 0.0
    end
    try
        pr = rampowerdraw(freeram) 
    catch ex
        println(ex.msg)
        return 0.0
    end
    endtime = time()
    elapsedtime = (endtime - starttime)*0.0002777778
    return 1.58*elapsedtime*(pc + pr + g*pg)/1000  
end
