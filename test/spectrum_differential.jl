using VoiceConversion
using Base.Test

using MCepAlpha
using WAV
using HDF5, JLD
import SPTK: MLSADF, synthesis!

# Load source speaker's (`clb`) speech signal.
wavpath = joinpath(Pkg.dir("VoiceConversion", "test", "data",
                           "clb_a0028.wav"))
x, fs = wavread(wavpath, format="int")
@assert size(x, 2) == 1 "The input data must be monoral."
x = float64(x[:])
fs = int(fs)
period = 5.0
order = 40
alpha = mcepalpha(fs)

# Mel-cepstrum extraction based on WORLD.
src_clb28 = world_mcep(x, fs, period, order, alpha)
@test !any(isnan(src_clb28))

# Female (`clb`) to female (`slt`) voice conversion demo based on
# Statistical Voice Conversion based on Spectrum Differential [Kobayashi 2014]
# frame-by-frame mapping
function spectrum_differetial_clb_to_slt()
    src = copy(src_clb28)

    # Load GMM to convert speech signal of `clb` to that of `slt`,
    # which is trained on CMU Arctic speech database.
    # mixture: 16, order of mel-cepstrum: 40
    modelpath = joinpath(Pkg.dir("VoiceConversion"), "test", "model",
                         "clb_to_slt_gmm32_order40_diff.jld")
    gmm = load(modelpath)
    @assert gmm["diff"]

    # Construct frame-by-frame GMM parameter mapping
    mapper = GMMMap(gmm)

    # Perform parameter conversion
    converted = vc(mapper, src)
    @test !any(isnan(converted))

    # remove power coef. in the original signal
    converted[1,:] = 0.0

    # Waveform modification
    mf = MLSADF(order)
    hopsize = int(fs / (1000 / period))
    synthesized = synthesis!(mf, x, converted, alpha, hopsize)
    @test !any(isnan(converted))
end

# Female (`clb`) to female (`slt`) voice conversion demo based on
# Statistical Voice Conversion based on Spectrum Differential [Kobayashi 2014]
# trajectory-based paramter mapping
function spectrum_differetial_trajectory_clb_to_slt()
    src = copy(src_clb28)
    
    # add dynamic feature
    src = [src[1,:], push_delta(src[2:end,:])]

    # Load GMM to convert speech signal of `clb` to that of `slt`,
    # which is trained on CMU Arctic speech database.
    # mixture: 16, order of mel-cepstrum: 40+40 (with dynamic feature)
    modelpath = joinpath(Pkg.dir("VoiceConversion"), "test", "model",
                         "clb_to_slt_gmm32_order40_diff_delta.jld")
    gmm = load(modelpath)
    @assert gmm["diff"]

    # Construct trajectory-based GMM parameter mapping
    mapper = TrajectoryGMMMap(GMMMap(gmm), 70)

    # Perform parameter conversion
    converted = vc(mapper, src)
    @test !any(isnan(converted))

    # remove power coef. in the original signal
    converted[1,:] = 0.0

    # Waveform modification
    mf = MLSADF(order)
    hopsize = int(fs / (1000 / period))
    synthesized = synthesis!(mf, x, converted, alpha, hopsize)
    @test !any(isnan(converted))
end

spectrum_differetial_clb_to_slt()
spectrum_differetial_trajectory_clb_to_slt()
