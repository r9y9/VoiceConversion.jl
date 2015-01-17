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

using VoiceConversion.Tools

using Logging
@Logging.configure(level=DEBUG, output=STDOUT)

searchdir(path, key) = filter(x -> contains(x, key), readdir(path))

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
        dstpath = joinpath(dstdir, string(splitext(basename(path))[1], "_wmcep.jld"))

        @info("Start processing $(path)")
        elapsed = @elapsed wmcep(path, period, order, α, dstpath)
        @info("Elapsed time in feature extraction is $(elapsed) sec.")
        @info("Dumped to $(dstpath)")

        count += 1
        count >= nmax && break
    end

    println("Finished")
end
