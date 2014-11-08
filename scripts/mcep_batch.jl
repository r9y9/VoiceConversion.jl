using DocOpt

doc="""Mel-cepstrum extraction for audio signals in batch.
Usage:
    mcep_batch.jl [options] <src_dir> <dst_dir>
    mcep_batch.jl --version
    mcep_batch.jl -h | --help

Options:
    -h --help        show this message
    --period=PERIOD  frame period in msec [default: 5.0]
    --order=ORDER    order of mel cepsrum [default: 40]
    --alpha=ALPHA    all-pass constant [default: 0.0]
    --max=MAX        Maximum number that will be processed [default: 100]
"""

using VoiceConversion
using MCepAlpha
using WORLD: get_fftsize_for_cheaptrick
using WAV
using HDF5, JLD

searchdir(path, key) = filter(x -> contains(x, key), readdir(path))

function main()
    args = docopt(doc, version=v"0.0.1")

    srcdir = args["<src_dir>"]
    dstdir = args["<dst_dir>"]
    if !isdir(dstdir)
        info("Create $(dstdir)")
        mkdir(dstdir)
    end

    const period = float(args["--period"])
    const order = int(args["--order"])
    const nmax = int(args["--max"])
    alpha = float(args["--alpha"])

    files = searchdir(srcdir, ".wav")
    info("$(length(files)) data found.")

    count = 0
    for filename in files
        path = joinpath(srcdir, filename)
        x, fs = wavread(path, format="int")
        @assert size(x, 2) == 1 "The input data must be monoral."
        x = float64(x[:])
        fs = int(fs)

        if alpha == 0.0
            alpha = mcepalpha(fs)
        end

        info("Processing $(path)")        
        mcgram = world_mcep(x, fs, period, order, alpha)

        dstpath = joinpath(dstdir, string(splitext(basename(path))[1], ".jld"))
        save(dstpath,
             "description", "WORLD-based Mel-cepstrum",
             "period", period,
             "fs", fs,
             "framelen", get_fftsize_for_cheaptrick(fs),
             "order", order,
             "alpha", alpha,
             "feature_matrix", mcgram,
             "jl-version", VERSION
             )
        info("Dumped to $(dstpath)")

        count += 1
        if count >= nmax
            break
        end
    end

    println("Finished")
end

main()
