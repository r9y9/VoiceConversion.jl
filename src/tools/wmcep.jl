# WORLD-based mel-cesptrum extraction.
function _wmcep(x::AbstractVector,
                fs::Integer,
                period::FloatingPoint,
                order::Integer,
                α::FloatingPoint;
                f0refine::Bool=true,
                )
    w = World(fs, period)

    # Fundamental frequency (f0) estimation by DIO
    f0, timeaxis = dio(w, x)

    # F0 re-estimation by StoneMask
    if f0refine
        f0 = stonemask(w, x, timeaxis, f0)
    end

    # Spectral envelope estimation
    spectrogram = cheaptrick(w, x, timeaxis, f0)

    # Spectral envelop -> Mel-cesptrum
    mc = wsp2mc(spectrogram, order, α)

    mc
end

function wmcep_save(mc::AbstractMatrix,
                    fs::Integer,
                    period::FloatingPoint,
                    order::Int,
                    α::FloatingPoint,
                    savepath)
    save(savepath,
         "description", "WORLD-based Mel-cepstrum",
         "type", "MelCepstrum",
         "fs", fs,
         "period", period,
         "order", order,
         "fftlen", get_fftsize_for_cheaptrick(fs),
         "alpha", α,
         "feature_matrix", mc
         )
end

function wmcep(wavpath, # filepath for the target wav file
               period::FloatingPoint,
               order::Integer,
               α::FloatingPoint,
               savepath;
               autoalpha::Bool=true,
               f0refine::Bool=true)
    x, fs = wavread(wavpath)
    size(x, 2) != 1 && error("The input data must be monoral.")
    x = vec(x)

    if autoalpha
        α = mcepalpha(fs)
    end

    mc = _wmcep(x, fs, period, order, α; f0refine=f0refine)
    wmcep_save(mc, fs, period, order, α, savepath)
end
