# Statistical Voice Conversion based on Spectrum Differential [Kobayashi 2014]

### basic setup

# Load source speaker's (`clb`) speech signal.
path = joinpath(Pkg.dir("VoiceConversion", "test", "data", "clb_a0028.txt"))
x = vec(readdlm(path))
fs = 16000
period = 5.0
order = 40
alpha = mcepalpha(fs)

# Mel-cepstrum extraction based on WORLD.
f0, timeaxis = dio(x, fs, DioOption(period=period))
f0 = stonemask(x, fs, timeaxis, f0)
spectrogram = cheaptrick(x, fs, timeaxis, f0)
src_clb28 = sp2mc(spectrogram, order, alpha)
@test !any(isnan(src_clb28))

x_clb28 = copy(x)

function diffvc_base(src, mapper)
    # Perform parameter conversion
    converted = vc(mapper, src)
    @test !any(isnan(converted))

    # remove power coef. in the converted signal
    converted[1,:] = 0.0

    # Waveform modification
    mf = MLSADF(order, alpha)
    hopsize = convert(Int, round(fs / (1000 / period)))
    synthesis!(mf, x_clb28, mc2b(converted, alpha), hopsize)
end

println("testing: voice conversion based on direct waveform modification")

let
    println("Female (`clb`) to female (`slt`) voice conversion")
    println("GMM-based frame-by-frame mapping")
    x = copy(src_clb28)

    # Load GMM to convert speech signal of `clb` to that of `slt`,
    # mixture: 32, order of mel-cepstrum: 40
    modelpath = joinpath(Pkg.dir("VoiceConversion"), "models",
                         "clb_to_slt_gmm32_order40_diff.jld")
    gmm = load(modelpath)
    @assert gmm["diff"]

    # Construct GMM-based frame-by-frame mapping
    mapper = GMMMap(gmm["weights"], gmm["means"], gmm["covars"])

    y = diffvc_base(x, mapper)
    @test !any(isnan(y))
end

let
    println("Female (`clb`) to female (`slt`) voice conversion")
    println("GMM-based trajectory paramter mapping")
    x = copy(src_clb28)

    # add dynamic feature
    x = [x[1,:], push_delta(x[2:end,:])]

    # Load GMM to convert speech signal of `clb` to that of `slt`,
    # mixture: 32, order of mel-cepstrum: 40+40 (with delta feature)
    modelpath = joinpath(Pkg.dir("VoiceConversion"), "models",
                         "clb_to_slt_gmm32_order40_diff_with_delta.jld")
    gmm = load(modelpath)
    @assert gmm["diff"]

    # Construct trajectory-based GMM parameter mapping
    mapper = GMMMap(gmm["weights"], gmm["means"], gmm["covars"])
    mapper = TrajectoryGMMMap(mapper, 70)

    y = diffvc_base(x, mapper)
    @test !any(isnan(y))
end
