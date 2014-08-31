using WORLD

import SPTK
const sptk = SPTK

# logamp2mcep converts log-amplitude spectrum to mel-cepstrum.
function logamp2mcep(logamp::Vector{Float64}, order::Int, alpha::Float64)
    ceps = real(ifft(logamp))
    ceps[1] /= 2.0
    return sptk.freqt(ceps, order, alpha)
end

# mcep2e computes energy from mel-cepstrum.
function mcep2e(mc::Vector{Float64}, alpha::Float64, len::Int)
    # back to linear frequency domain
    c = sptk.freqt(mc, len-1, -alpha)

    # compute impule response from cepsturm
    ir = sptk.c2ir(c, len)

    return sumabs2(ir)
end

mcep2e(mat::Matrix{Float64}, alpha, len) =
    [mcep2e(mat[:,i], alpha, len) for i=1:size(mat, 2)]

# world_mcep computes mel-cepstrum for whole input signal using
# WORLD-based spectral envelope estimation.
function world_mcep(x, fs, period::Float64=5.0, order::Int=25,
                    alpha::Float64=0.35)
    w = World(fs=fs, period=period)

    # Fundamental frequency (f0) estimation by DIO
    f0, timeaxis = dio1(w, x)

    # F0 re-estimation by StoneMask
    f0 = stonemask(w, x, timeaxis, f0)

    # Spectral envelope estimation
    spectrogram = cheaptrick(w, x, timeaxis, f0)

    # Spectral envelop -> Mel-cesptrum
    mcgram = zeros(order+1, size(spectrogram, 2))
    for i=1:size(spectrogram, 2)
        spec = spectrogram[:,i]
        symmetrized = [spec, reverse(spec[2:end])]
        @assert length(symmetrized) == length(spec)*2-1
        logspec = log(symmetrized)
        mcgram[:,i] = logamp2mcep(logspec, order, alpha)
    end

    return mcgram
end
