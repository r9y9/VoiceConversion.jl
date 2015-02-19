using DocOpt

doc="""WORLD-based mel-cepstrum extraction for audio signals.
Usage:
    mcep.jl [options] <src_dir> <dst_dir>
    mcep.jl --version
    mcep.jl -h | --help

Options:
    -h --help        show this message
    --period=PERIOD  frame period in msec [default: 5.0]
    --order=ORDER    order of mel cepsrum [default: 40]
    --alpha=ALPHA    all-pass constant [default: 0.0]
    --max=MAX        Maximum number that will be processed [default: 200]
"""

using VoiceConversion
using WAV
using WORLD
using HDF5, JLD

import MelGeneralizedCepstrums: mcepalpha

using Logging
@Logging.configure(level=DEBUG, output=STDOUT)

searchdir(path, key) = filter(x -> contains(x, key), readdir(path))

function mkdir_if_not_exist(dir)
    if !isdir(dir)
        println("Create $(dir)")
        run(`mkdir -p $dir`)
    end
end

# process one file
function _mcep(path, period, order, α, savepath)
    x, fs = wavread(path)
    @assert size(x, 2) == 1 "The input data must be monoral."
    x = float(vec(x))
    fs = int(fs)

    if α == 0.0
        α = mcepalpha(fs)
    end

    w = World(fs, period)
    f0, timeaxis = dio(w, x)
    f0 = stonemask(w, x, timeaxis, f0)
    spectrogram = cheaptrick(w, x, timeaxis, f0)
    mc = sp2mc(spectrogram, order, α)

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

let
    args = docopt(doc, version=v"0.0.2")

    srcdir = args["<src_dir>"]
    dstdir = args["<dst_dir>"]
    mkdir_if_not_exist(dstdir)

    period = float(args["--period"])
    order = int(args["--order"])
    nmax = int(args["--max"])
    α = float(args["--alpha"])

    files = searchdir(srcdir, ".wav")
    @info("$(length(files)) data found.")

    # perform feature extraction for each file
    count = 0
    for filename in files
        path = joinpath(srcdir, filename)
        savepath = joinpath(dstdir, string(splitext(basename(path))[1], "_wmcep.jld"))

        @info("Start processing $(path)")
        elapsed = @elapsed _mcep(path, period, order, α, savepath)
        @info("Elapsed time in feature extraction is $(elapsed) sec.")
        @info("Dumped to $(savepath)")

        count += 1
        count >= nmax && break
    end

    println("Finished")
end
