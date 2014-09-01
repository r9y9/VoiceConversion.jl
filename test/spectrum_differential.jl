using VoiceConversion
using Base.Test

using MCepAlpha
using WAV
using HDF5, JLD
import SPTK: MLSADF, synthesis!

# Statistical Voice Conversion based on Spectrum Differential
# TODO(ryuichi) proper refer
function spectrum_differetial()
    wavpath = joinpath(Pkg.dir("VoiceConversion", "test", "data",
                               "clb_a0028.wav"))
    x, fs = wavread(wavpath, format="int")
    @assert size(x, 2) == 1 "The input data must be monoral."
    x = float64(x[:])
    const fs = int(fs)
    const period = 5.0
    const order = 40
    const alpha = mcepalpha(fs)

    # Feature extraction that will be converted
    src = world_mcep(x, fs, period, order, alpha)
    @assert !any(isnan(src))

    # Load mapping GMM (mixture: 16, order of mel-cepstrum: 40)
    # clb to slt from CMU Arctic speech database.
    modelpath = joinpath(Pkg.dir("VoiceConversion"), "test", "model",
                         "clb_to_slt_gmm16_order40_diff.jld")
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

spectrum_differetial()
