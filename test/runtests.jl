using GreenFlux
using Test
using Flux

@testset "GreenFlux.jl" begin
    #convol = Conv((15,15),1=>2,tand)
    #dense = Dense(23,31,gelu)
    #maxpoo = MaxPool((12,65)) 
    # TODO: GlobalMaxPool
    #mpool = MeanPool((3,3)) 
    # TODO: GlobalMeanPool
    #dconv = DepthwiseConv((21,21),6=>12,relu)
    #ctrans = ConvTranspose((7,7),2=>4,tan) 
    #cc = CrossCor((2, 2), 1=>16, relu6)
    #gr = GRU(4,8) 
    #lst = LSTM(3,3)
    #rn = RNN(3,6) 
    #maxo = Maxout(()->Dense(35, 27), 4)
    #@test_throws GreenFlux.NoNvidiaSMI GreenFlux.gpupowerdraw()
    #@test_throws GreenFlux.NoPowerStat GreenFlux.cpupowerdraw()
    #@test typeof(GreenFlux.rampowerdraw()) <: Float64
    #@test typeof(avgpowerdraw()) <: Float64  
    #@test typeof(GreenFlux.layerflops(convol,(2,2))) == Tuple{Float64,Tuple{Int64,Int64}}
    #@test typeof(GreenFlux.layerflops(dense,(4,4))) == Tuple{Float64,Tuple{Int64,Int64}}
    #@test typeof(GreenFlux.layerflops(maxpoo,(9,9))) == Tuple{Float64,Tuple{Int64,Int64}}
    #@test typeof(GreenFlux.layerflops(mpool,(2,2))) == Tuple{Float64,Tuple{Int64,Int64}}
    #@test typeof(GreenFlux.layerflops(dconv,(1,1))) == Tuple{Float64,Tuple{Int64,Int64}}
    #@test typeof(GreenFlux.layerflops(ctrans,(6,6))) == Tuple{Float64,Tuple{Int64,Int64}}
    #@test typeof(GreenFlux.layerflops(cc,(3,3))) == Tuple{Float64,Tuple{Int64,Int64}}
    #@test typeof(GreenFlux.layerflops(gr,(77,77))) == Tuple{Float64,Tuple{Int64,Int64}}
    #@test typeof(GreenFlux.layerflops(lst,(8,8))) == Tuple{Float64,Tuple{Int64,Int64}}
    #@test typeof(GreenFlux.layerflops(rn,(4,4))) == Tuple{Float64,Tuple{Int64,Int64}}
    #@test typeof(GreenFlux.layerflops(maxo,(5,5))) == Tuple{Float64,Tuple{Int64,Int64}}
end
