using GreenFlux
using Test
using Flux

@testset "GreenFlux.jl" begin
    convol = Conv((15,15),4=>2,tanh)
    dense = Dense(23,31,gelu)
    maxpoo = MaxPool((12,65)) 
    # TODO: GlobalMaxPool
    mpool = MeanPool((3,3)) 
    # TODO: GlobalMeanPool
    dconv = DepthwiseConv((21,21),6=>12,relu)
    ctrans = ConvTranspose((7,7),2=>4,identity) 
    cc = CrossCor((2, 2), 1=>16, relu6)
    gr = GRU(4,8) 
    lst = LSTM(3,3)
    rn = RNN(3,6) 
    #maxo = Maxout(()->Dense(35, 27), 4)
    maxo = Maxout((12,12))
    @test_throws Base.IOError GreenFlux.gpupowerdraw()
    @test_throws Base.IOError GreenFlux.cpupowerdraw()
    @test_throws Base.IOError GreenFlux.rampowerdraw()
    if Sys.islinux()
        @test typeof(avgpowerdraw()) <: Float64  
    end
    @test GreenFlux.layerflops(convol,(28,28)) == (354368.0, (14, 14, 2))
    @test GreenFlux.layerflops(dense,(4,4)) == (2304.0, (31, 1))
    @test GreenFlux.layerflops(maxpoo,(100,100)) == (6240.0, (8, 1)) 
    @test GreenFlux.layerflops(mpool,(8,8)) == (36.0, (2, 2)) 
    @test GreenFlux.layerflops(dconv,(30,30)) == (177600.0, (10, 10, 6)) 
    @test GreenFlux.layerflops(ctrans,(6,6)) == (56736.0, (12, 12, 2))
    @test GreenFlux.layerflops(cc,(3,3)) == (224.0, (2, 2, 16))
    @test GreenFlux.layerflops(gr,(77,77)) == (6099.0, (24, 1)) 
    @test GreenFlux.layerflops(lst,(8,8)) == (1968.0, (12, 1))
    @test GreenFlux.layerflops(rn,(4,4)) == (546.0, (6, 1))
    @test GreenFlux.layerflops(maxo,(5,5)) == (3600.0, (5, 1))
end
