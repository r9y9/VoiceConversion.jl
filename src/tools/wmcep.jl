# WORLD-based mel-cesptrum extraction.
function wmcep(x::AbstractVector,
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

function save_wmcep(savepath,
                    mc::AbstractMatrix,
                    fs::Integer,
                    period::FloatingPoint,
                    order::Int,
                    α::FloatingPoint)
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
