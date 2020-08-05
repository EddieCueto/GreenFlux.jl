function gotocompile(dev::Bool=false)
    localpaths = DEPOT_PATH
    if !dev
        comp_path = localpaths[1]*"/packages/GreenFlux/"
        cd(comp_path)
        comp_path = comp_path*readdir()[1]
        cd(comp_path)
        return comp_path
    elseif dev
        comp_path = localpaths[1]*"/dev/GreenFlux/"
        cd(comp_path)
        return comp_path
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