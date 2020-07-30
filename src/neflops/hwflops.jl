
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
    cpuusage = pipeline(`top`, `grep \%Cpu`, "./src/neflops/cpuper")
    currenttime = Dates.Time(Dates.now())
    firstrun = 1
    while (Dates.Time(Dates.now()) - currenttime).value / 1000000000.0 < 60.0
        if firstrun == 1
            asyresult = run(cpuusage,wait = false)
            #println("I entered here once......")
            #println((Dates.Time(Dates.now()) - currenttime).value / 1000000000.0)
        end
        firstrun += 1
    end
end