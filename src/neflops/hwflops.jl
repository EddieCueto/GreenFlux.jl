
function getcpuclock()
    getclkspd = `lscpu`
    cpuclockspeed = read(getclkspd, String);
    cpuclockspeed = split(cpuclockspeed, "\n")
    cpuclockspeed = split(cpuclockspeed[15], " ")
    cpuclockspeed = filter(x->x≠"",cpuclockspeed)
    cpuclockspeed = parse(Float64,cpuclockspeed[3])
    return cpuclockspeed
end


function getcpuusage()
    tempfile = "./src/neflops/cpuper"
    cpuusage = pipeline(`top`, `grep \%CPU`, tempfile)
    ch = Channel(1)
    put!(ch,@async run(cpuusage))
    sleep(60)
    if isready(ch)
        take!(ch)
    end
    data = open(tempfile) do text
        read(text, String)
    end
    rm(tempfile)
    data = split(data, "\n")
    return data
end