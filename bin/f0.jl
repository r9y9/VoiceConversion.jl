using DocOpt

doc="""WORLD-based F0 extraction for audio signals.
Usage:
    f0.jl [options] <src_dir> <dst_dir>
    f0.jl --version
    f0.jl -h | --help

Options:
    -h --help        show this message
    --period=PERIOD  frame period in msec [default: 5.0]
    --max=MAX        Maximum number that will be processed [default: 200]
"""

# TODO(ryuichi) allow to specify DioOption as command line options?

using VoiceConversion.Tools
using WAV

using Logging
@Logging.configure(level=DEBUG, output=STDOUT)

searchdir(path, key) = filter(x -> contains(x, key), readdir(path))

let
    args = docopt(doc, version=v"0.0.2")

    srcdir = args["<src_dir>"]
    dstdir = args["<dst_dir>"]
    mkdir_if_not_exist(dstdir)

    period = parse(Float64, args["--period"])
    nmax = parse(Int, args["--max"])

    files = searchdir(srcdir, ".wav")
    @info("$(length(files)) data found.")

    # perform feature extraction for each file
    for (count, filename) in enumerate(files)
        path = joinpath(srcdir, filename)
        savepath = joinpath(dstdir, string(splitext(basename(path))[1],
                                          "_f0.jld"))

        @info("Start processing $(path)")
        elapsed = @elapsed begin
            x, fs = wavread(path)
            size(x, 2) != 1 && error("The input data must be monoral.")
            x = map(Float64, vec(x))

            f0, timeaxis = dio(x, fs, DioOption(period=period))
            f0 = stonemask(x, fs, timeaxis, f0)

            save(savepath,
                 "description", "WORLD-based F0",
                 "type", "f0",
                 "fs", fs,
                 "period", period,
                 "feature_vector", f0
                 )
        end
        @info("Elapsed time in F0 estimation is $(elapsed) sec.")
        @info("Dumped to $(savepath)")

        count += 1
        count > nmax && break
    end

    println("Finished")
end
