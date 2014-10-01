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
    --order=ORDER    order of mel cepsrum [default: 40]
    --alpha=ALPHA    all-pass constant [default: 0.0]
"""

using VoiceConversion
using MCepAlpha
using WORLD: get_fftsize_for_cheaptrick
using WAV
using HDF5, JLD

function main()
    args = docopt(doc, version=v"0.0.1")

    x, fs = wavread(args["<input_audio>"], format="int")
    @assert size(x, 2) == 1 "The input data must be monoral."
    x = float64(x[:])
    fs = int(fs)

    const period = float(args["--period"])
    const order = int(args["--order"])
    alpha = float(args["--alpha"])
    if alpha == 0.0
        alpha = mcepalpha(fs)
    end

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
