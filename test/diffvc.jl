# Statistical Voice Conversion based on Spectrum Differential [Kobayashi 2014]

### basic setup

# Load source speaker's (`clb`) speech signal.
wavpath = joinpath(Pkg.dir("VoiceConversion", "test", "data",
                           "clb_a0028.wav"))
x, fs = wavread(wavpath)
@assert size(x, 2) == 1 "The input data must be monoral."
x = float(vec(x))
fs = int(fs)
period = 5.0
order = 40
alpha = mcepalpha(fs)

# Mel-cepstrum extraction based on WORLD.
src_clb28 = world_mcep(x, fs, period, order, alpha)
@test !any(isnan(src_clb28))

function diffvc_base(src, mapper)
    # Perform parameter conversion
    converted = vc(mapper, src)
    @test !any(isnan(converted))

    # remove power coef. in the converted signal
    converted[1,:] = 0.0

    # Waveform modification
    mf = MLSADF(order, alpha)
    hopsize = int(fs / (1000 / period))
    synthesis!(mf, x, converted, hopsize)
end

# Female (`clb`) to female (`slt`) voice conversion demo
# frame-by-frame mapping
function diffvc_clb2slt()
    x = copy(src_clb28)

    # Load GMM to convert speech signal of `clb` to that of `slt`,
    # mixture: 32, order of mel-cepstrum: 40
    modelpath = joinpath(Pkg.dir("VoiceConversion"), "test", "model",
                         "clb_to_slt_gmm32_order40_diff.jld")
    gmm = load(modelpath)
    @assert gmm["diff"]

    # Construct GMM-based frame-by-frame mapping
    mapper = GMMMap(gmm["weights"], gmm["means"], gmm["covars"])

    y = diffvc_base(x, mapper)
    @test !any(isnan(y))
end

# Female (`clb`) to female (`slt`) voice conversion demo
# trajectory-based paramter mapping
function trajectory_diffvc_clb2slt()
    x = copy(src_clb28)
    
    # add dynamic feature
    x = [x[1,:], push_delta(x[2:end,:])]

    # Load GMM to convert speech signal of `clb` to that of `slt`,
    # mixture: 32, order of mel-cepstrum: 40+40 (with delta feature)
    modelpath = joinpath(Pkg.dir("VoiceConversion"), "test", "model",
                         "clb_to_slt_gmm32_order40_diff_with_delta.jld")
    gmm = load(modelpath)
    @assert gmm["diff"]

    # Construct trajectory-based GMM parameter mapping
    mapper = GMMMap(gmm["weights"], gmm["means"], gmm["covars"])
    mapper = TrajectoryGMMMap(mapper, 70)

    y = diffvc_base(x, mapper)
    @test !any(isnan(y))
end

### Tests

println("testing: voice conversion process based on direct waveform modification")
diffvc_clb2slt()
trajectory_diffvc_clb2slt()
