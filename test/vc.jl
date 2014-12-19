# Voice Conversion

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

w = World(fs=fs, period=period)

# Fundamental frequency (f0) estimation by DIO
f0, timeaxis = dio(w, x)

# F0 re-estimation by StoneMask
f0 = stonemask(w, x, timeaxis, f0)

# Spectral envelope estimation
spectrogram = cheaptrick(w, x, timeaxis, f0)

# Spectral envelop -> Mel-cesptrum
src_clb28 = wsp2mc(spectrogram, order, alpha)
@test !any(isnan(src_clb28))

# aperiodicity ratio estimation
ap = aperiodicityratio(w, x, f0, timeaxis)

x_clb28 = copy(x)

# peform conversion and return synthesized waveform
function vc_base(src, mapper)
    converted = vc(mapper, src)
    @test !any(isnan(converted))
    converted_spectrogram = mc2wsp(converted, size(spectrogram,1), -alpha)
    synthesis_from_aperiodicity(w, f0, converted_spectrogram, ap, length(x_clb28))
end

# Female (`clb`) to female (`slt`) voice conversion demo
# frame-by-frame mapping
function vc_clb2slt()
    x = copy(src_clb28)

    # Load GMM to convert speech signal of `clb` to that of `slt` and vise versa,
    # mixture: 32, order of mel-cepstrum: 40
    modelpath = joinpath(Pkg.dir("VoiceConversion"), "test", "model",
                         "clb_and_slt_gmm32_order40.jld")
    gmm = load(modelpath)
    @assert !gmm["diff"]

    # Construct GMM-based frame-by-frame mapping
    mapper = GMMMap(gmm["weights"], gmm["means"], gmm["covars"])

    y = vc_base(x, mapper)
    @test !any(isnan(y))
end

# Female (`clb`) to female (`slt`) voice conversion demo
# trajectory-based paramter mapping
function trajectory_vc_clb2slt()
    x = copy(src_clb28)

    # add dynamic feature
    x = [x[1,:], push_delta(x[2:end,:])]

    # Load GMM to convert speech signal of `clb` to that of `slt`,
    # mixture: 32, order of mel-cepstrum: 40+40 (with delta feature)
    modelpath = joinpath(Pkg.dir("VoiceConversion"), "test", "model",
                         "clb_and_slt_gmm32_order40_with_delta.jld")
    gmm = load(modelpath)
    @assert !gmm["diff"]

    # Construct trajectory-based GMM parameter mapping
    mapper = GMMMap(gmm["weights"], gmm["means"], gmm["covars"])
    mapper = TrajectoryGMMMap(mapper, 70)
    y = vc_base(x, mapper)
    @test !any(isnan(y))
end

### Tests

println("testing: voice conversion process using WORLD vocoder.")
vc_clb2slt()
trajectory_vc_clb2slt()
