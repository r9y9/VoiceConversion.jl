# Mel-cesptrum related functions

# wsp2mc peforms conversion from WORLD-based spectral envelope to mel-cepstrum
function wsp2mc(spec::AbstractVector{Float64}, order::Int, α::Float64;
                fftlen::Int=(length(spec)-1)*2)
    logspec = log(spec)

    # transform to cepstrum domain
    c = real(irfft(logspec, fftlen))
    c[1] /= 2.0

    # frequency to mel domain
    freqt(c, order, α)
end

# mc2wsp performs conversion from mel-cepstrum to WORLD-based spectral envelope
function mc2wsp(mc::Vector{Float64}, fftlen::Int, α::Float64)
    # back to cepstrum from mel-cesptrum
    c = freqt(mc, fftlen>>1, -α)
    c[1] *= 2.0

    symc = zeros(eltype(mc), fftlen)
    copy!(symc, c)
    for i=2:length(c)
        @inbounds symc[end-i+1] = c[i]
    end

    # back to spectrum
    exp(real(rfft(symc)))
end

# extend vector to vector transformation for matrix input
for f in [:wsp2mc,
          :mc2wsp,
          ]
    @eval begin
        function $f(x::AbstractMatrix{Float64}, args...; kargs...)
            r = $f(x[:, 1], args...; kargs...)
            ret = Array(eltype(r), size(r, 1), size(x, 2))
            for i = 1:length(r)
                @inbounds ret[i, 1] = r[i]
            end
            for i = 2:size(x, 2)
                @inbounds ret[:, i] = $f(x[:, i], args...; kargs...)
            end
            ret
        end
    end
end

# world_mcep computes mel-cepstrum for whole input signal using WORLD-based
# spectral envelope estimation.
# will be deprecated
function world_mcep(x, fs, period::Float64=5.0, order::Int=25,
                    α::Float64=0.35)
    w = World(fs, period)

    # Fundamental frequency (f0) estimation by DIO
    f0, timeaxis = dio(w, x)

    # F0 re-estimation by StoneMask
    f0 = stonemask(w, x, timeaxis, f0)

    # Spectral envelope estimation
    spectrogram = cheaptrick(w, x, timeaxis, f0)

    # Spectral envelop -> Mel-cesptrum
    mcgram = wsp2mc(spectrogram, order, α)

    mcgram
end
