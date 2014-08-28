using DocOpt

doc="""Mel-cepstrum extraction for audio signals using WORLD-based high
accurate spectral envelope estimation method.

Usage:
    mcep.jl [options] <input_audio> <dst>
    mcep.jl --version
    mcep.jl -h | --help

Options:
    -h --help        show this message
    --period=PERIOD  frame period in msec [default: 5.0]
    --order=ORDER    order of mel cepsrum [default: 25]
    --alpha=ALPHA    all-pass constant [default: 0.35]
"""

using VoiceConversion
using WORLD
using WAV
using HDF5, JLD

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
    mcgram = zeros(order+1, size(spectrogram, 1))
    for i=1:size(spectrogram, 1)
        spec = spectrogram[i,:][:]
        symmetrized = [spec, reverse(spec[2:end])]
        @assert length(symmetrized) == length(spec)*2-1
        logspec = log(symmetrized)
        mcgram[:,i] = logamp2mcep(logspec, order, alpha)
    end

    return mcgram
end

function main()
    args = docopt(doc, version=v"0.0.1")

    x, fs = wavread(args["<input_audio>"], format="int")
    @assert size(x, 2) == 1 "The input data must be monoral."
    x = float64(x[:])
    fs = int(fs)

    period = float(args["--period"])
    order = int(args["--order"])
    alpha = float(args["--alpha"])

    mcgram = world_mcep(x, fs, period, order, alpha)

    save(args["<dst>"],
         "description", "WORLD-based Mel-cepstrum",
         "period", period,
         "fs", fs,
         "framelen", get_fftsize_for_cheaptrick(fs),
         "order", order,
         "alpha", alpha,
         "feature_matrix", mcgram,
         )

    println("Dumped to ", args["<dst>"])
end

@time main()
