function test_gmmmap()
    modelpath = joinpath(Pkg.dir("VoiceConversion"), "models",
                         "clb_to_slt_gmm32_order40_diff.jld")
    gmm = load(modelpath)
    @assert gmm["diff"]

    mapper = GMMMap(gmm["weights"], gmm["means"], gmm["covars"])
    @test length(mapper) == 1

    const D = div(size(gmm["means"], 1), 2)
    @test dim(mapper) == D
    @test ncomponents(mapper) == length(gmm["weights"])
    @test size(mapper) == (D, 1)
end

println("testing: GMM-based parameter conversion")
test_gmmmap()
