using GreenFlux
using Test
using Flux

@testset "GreenFlux.jl" begin
    convol = Conv((15,15),1=>2,tand)
    dense = Dense(23,31,gelu)
    @test_throws GreenFlux.NoNvidiaSMI avgpowerdraw()
    @test_throws GreenFlux.NoPowerStat avgpowerdraw()
    @test typeof(avgpowerdraw()) <: Float64  
end
