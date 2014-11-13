function test_gmmmap()
    modelpath = joinpath(Pkg.dir("VoiceConversion"), "test", "model",
                         "clb_to_slt_gmm32_order40_diff.jld")
    gmm = load(modelpath)
    @assert gmm["diff"]
    
    mapper = GMMMap(gmm)
    @test length(mapper) == 1

    const D = div(size(gmm["means"], 1), 2)
    @test dim(mapper) == D
    @test ncomponents(mapper) == length(gmm["weights"])
end

test_gmmmap()
