using DocOpt

doc="""Align two feature vector sequences and create parallel data in batch.
Note that filename of source and target feature is assumed to be same.

Usage:
    align.jl [options] <src_dir> <tgt_dir> <dst_dir>
    align.jl --version
    align.jl -h | --help

Options:
    -h --help       show this message
    --threshold=TH  threshold that is used to remove silence [default: -14.0]
    --max=MAX       Maximum number that will be processed [default: 200]
"""

using VoiceConversion.Tools
using HDF5, JLD

using Logging
@Logging.configure(level=DEBUG, output=STDOUT)

searchdir(path, key) = filter(x -> contains(x, key), readdir(path))

let
    args = docopt(doc, version=v"0.0.1")

    srcdir = args["<src_dir>"]
    tgtdir = args["<tgt_dir>"]
    dstdir = args["<dst_dir>"]
    mkdir_if_not_exist(dstdir)

    threshold = float(args["--threshold"])
    nmax = int(args["--max"])

    files = searchdir(srcdir, ".jld")
    @info("$(length(files)) data found.")

    count = 0
    for filename in files
        # filename is assumed to be same between src and tgt
        srcpath = joinpath(srcdir, filename)
        tgtpath = joinpath(tgtdir, filename)
        savepath = joinpath(dstdir, string(splitext(basename(srcpath))[1],
                                      "_parallel.jld"))

        @info("Start processing $(srcpath) and $(tgtpath)")
        elapsed = @elapsed begin
            src = load(srcpath)
            tgt = load(tgtpath)
            src, tgt = align!(src, tgt, threshold=threshold)
            save_align(savepath, src, tgt)
        end
        @info("Elapsed time in alignment is $(elapsed) sec.")
        @info("Dumped to $(savepath)")

        count += 1
        count >= nmax && break
    end

    @info("Finished")
end
