# Mel-cesptrum related functions

using WORLD

import SPTK: freqt, c2ir

# logamp2mc converts log-amplitude spectrum to mel-cepstrum.
function logamp2mc(logamp::Vector{Float64}, order::Int, alpha::Float64)
    ceps = real(ifft(logamp))
    ceps[1] /= 2.0
    freqt(ceps, order, alpha)
end

# mc2logamp performs mel-cepstrum to log amplitude conversion
function mc2logamp(mc::Vector{Float64}, freqbins::Int, alpha::Float64)
    ceps = freqt(mc, length(mc)-1, alpha)
    ceps[1] *= 2.0

    # adjast number of frequency bins and symmetrize
    actualceps = zeros(eltype(mc), freqbins)
    actualceps[1] = ceps[1]
    for i=2:length(ceps)
        actualceps[i] = ceps[i]
        actualceps[freqbins-i+2] = ceps[i]
    end
    real(fft(actualceps))
end

# mc2e computes energy from mel-cepstrum.
function mc2e(mc::Vector{Float64}, alpha::Float64, len::Int)
    # back to linear frequency domain
    c = freqt(mc, len-1, -alpha)

    # compute impule response from cepsturm
    ir = c2ir(c, len)

    sumabs2(ir)
end

mc2e(mat::Matrix{Float64}, alpha, len) =
    [mc2e(mat[:,i], alpha, len) for i=1:size(mat, 2)]

# world_mcep computes mel-cepstrum for whole input signal using WORLD-based
# spectral envelope estimation.
function world_mcep(x, fs, period::Float64=5.0, order::Int=25,
                    alpha::Float64=0.35)
    w = World(fs=fs, period=period)

    # Fundamental frequency (f0) estimation by DIO
    # TODO(ryuichi) replace dio1 to dio
    f0, timeaxis = dio1(w, x)

    # F0 re-estimation by StoneMask
    f0 = stonemask(w, x, timeaxis, f0)

    # Spectral envelope estimation
    spectrogram = cheaptrick(w, x, timeaxis, f0)

    # Spectral envelop -> Mel-cesptrum
    mcgram = wsp2mc(spectrogram, order, alpha)

    mcgram
end

# TODO(ryuichi): rename to sp2mc
# wsp2mc peforms conversion from WORLD-based spectral envelope to mel-cepstrum
function wsp2mc(spec::Vector{Float64}, order::Int, alpha::Float64)
    symmetrized = [spec, reverse(spec[2:end-1])]
    @assert length(symmetrized) == (length(spec)-1)*2
    logspec = log(symmetrized)
    logamp2mc(logspec, order, alpha)
end

function wsp2mc(spectrogram::Matrix{Float64}, order::Int, alpha::Float64)
    const T = size(spectrogram, 2)
    mcgram = zeros(order+1, T)
    for i=1:T
        mcgram[:,i] = wsp2mc(spectrogram[:,i], order, alpha)
    end
    mcgram
end

# TODO(ryuichi): rename to mc2sp
# mc2wsp performs conversion from mel-cepstrum to WORLD-based spectral envelope
function mc2wsp(mc::Vector{Float64}, freqbins::Int, alpha::Float64)
    const symmetrized_len = (freqbins-1)*2
    logamp = mc2logamp(mc, symmetrized_len, alpha)
    exp(logamp[1:freqbins])
end

function mc2wsp(mcgram::Matrix{Float64}, freqbins::Int, alpha::Float64)
    const T = size(mcgram, 2)
    spectrogram = Array(eltype(mcgram), freqbins, T)
    for t=1:T
        spectrogram[:,t] = mc2wsp(mcgram[:,t], freqbins, alpha)
    end
    spectrogram
end
