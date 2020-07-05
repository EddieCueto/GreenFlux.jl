struct NoNvidiaSMI <: Exception
    var::String
end

struct NoPowerStat <: Exception
    var::String
end

struct NoFree <: Exception
    var::String
end

Base.showerror(io::IO, e::NoNvidiaSMI) = print(io, e.var)
Base.showerror(io::IO, e::NoPowerStat) = print(io, e.var)
Base.showerror(io::IO, e::NoFree) = print(io, e.var)

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
                usagestr = "0.0"
            else
                usagestr = usagestr * g[5]
            end
            if g[7] == "N/A"
                capstr = "0.0"
            else
                capstr = capstr * g[7]
            end
            powerdraw = vcat(powerdraw, parse(Float64,usagestr))
            powercap = vcat(powercap, parse(Float64,capstr))
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
ratio of activated memory against the unactivated for the maximum power value and convert it 
to hours.
"""
function rampowerdraw()
    ramcommand = `free`
    powerused = Array{Float64}(undef,60)
    for count in 1:60
        ram = read(ramcommand, String);
        ram = split(ram,"\n")
        ram = split(ram[2]," ")
        filter!(x->x≠"",ram)
        usedram = parse(Float64,ram[3])
        totalram = parse(Float64,ram[2])
        powerused[count] = ((usedram*1.575)/totalram)*1.904
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
"""
function avgpowerdraw()
    starttime = time()
    g, pg, _ = gpupowerdraw()
    pc = cpupowerdraw()
    pr = rampowerdraw() 
    endtime = time()
    elapsedtime = (endtime - starttime)/3600
    return 1.58*elapsedtime*(pc + pr + g*pg)/1000  
end
