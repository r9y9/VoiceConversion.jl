using DocOpt

doc="""Mel-cepstrum extraction for audio signals in batch.
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
using MCepAlpha
using WORLD: get_fftsize_for_cheaptrick
using WAV
using HDF5, JLD

searchdir(path, key) = filter(x -> contains(x, key), readdir(path))

# process one file
function _mcep(path, period::Float64, order::Int, alpha::Float64, dstpath)
    x, fs = wavread(path)
    @assert size(x, 2) == 1 "The input data must be monoral."
    x = float(vec(x))
    fs = int(fs)

    if alpha == 0.0
        alpha = mcepalpha(fs)
    end

    mcgram = world_mcep(x, fs, period, order, alpha)
    
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
end

function main()
    args = docopt(doc, version=v"0.0.2")

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
        dstpath = joinpath(dstdir, string(splitext(basename(path))[1], ".jld"))

        info("Start processing $(path)")
        elapsed = @elapsed _mcep(path, period, order, alpha, dstpath)
        info("Elapsed time in feature extraction is $(elapsed) sec.")
        info("Dumped to $(dstpath)")

        count += 1
        if count >= nmax
            break
        end
    end

    println("Finished")
end

main()
