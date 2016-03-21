let
    println("testing: GMM-based parameter conversion")
    modelpath = joinpath(Pkg.dir("VoiceConversion"), "test", "models",
                         "clb_to_slt_gmm32_order40_diff.jld")
    gmm = load(modelpath)
    @assert gmm["diff"]

    mapper = GMMMap(gmm["weights"], gmm["means"], gmm["covars"])
    @test length(mapper) == 1

    D = div(size(gmm["means"], 1), 2)
    @test dim(mapper) == D
    @test ncomponents(mapper) == length(gmm["weights"])
    @test size(mapper) == (D, 1)
end
