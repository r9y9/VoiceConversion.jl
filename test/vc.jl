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

f0, timeaxis = dio(x, fs, DioOption(period=period))
f0 = stonemask(x, fs, timeaxis, f0)
spectrogram = cheaptrick(x, fs, timeaxis, f0)
src_clb28 = sp2mc(spectrogram, order, alpha)
@test !any(isnan(src_clb28))
ap = d4c(x, fs, timeaxis, f0)
@test !any(isnan(ap))

x_clb28 = copy(x)

# peform conversion and return synthesized waveform
function vc_base(src, mapper)
    converted = vc(mapper, src)
    @test !any(isnan(converted))
    fftlen = size(spectrogram,1)*2-1
    converted_spectrogram = mc2sp(converted, alpha, fftlen)
    synthesis(f0, converted_spectrogram, ap, period, fs, length(x_clb28))
end

println("testing: voice conversion using the WORLD vocoder.")

let
    println("Female (`clb`) to female (`slt`) voice conversion")
    println("GMM-based frame-by-frame mapping")
    x = copy(src_clb28)

    # Load GMM to convert speech signal of `clb` to that of `slt` and vise versa,
    # mixture: 32, order of mel-cepstrum: 40
    modelpath = joinpath(Pkg.dir("VoiceConversion"), "models",
                         "clb_and_slt_gmm32_order40.jld")
    gmm = load(modelpath)
    @assert !gmm["diff"]

    # Construct GMM-based frame-by-frame mapping
    mapper = GMMMap(gmm["weights"], gmm["means"], gmm["covars"])

    y = vc_base(x, mapper)
    @test !any(isnan(y))
end

let
    println("Female (`clb`) to female (`slt`) voice conversion")
    println("GMM-based trajectory parameter mapping")
    x = copy(src_clb28)

    # add dynamic feature
    x = [x[1,:], push_delta(x[2:end,:])]

    # Load GMM to convert speech signal of `clb` to that of `slt`,
    # mixture: 32, order of mel-cepstrum: 40+40 (with delta feature)
    modelpath = joinpath(Pkg.dir("VoiceConversion"), "models",
                         "clb_and_slt_gmm32_order40_with_delta.jld")
    gmm = load(modelpath)
    @assert !gmm["diff"]

    # Construct trajectory-based GMM parameter mapping
    mapper = GMMMap(gmm["weights"], gmm["means"], gmm["covars"])
    mapper = TrajectoryGMMMap(mapper, 70)
    y = vc_base(x, mapper)
    @test !any(isnan(y))
end
