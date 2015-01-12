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

using VoiceConversion
using HDF5, JLD

using Logging
@Logging.configure(level=DEBUG, output=STDOUT)

searchdir(path, key) = filter(x -> contains(x, key), readdir(path))

function _align(srcpath, tgtpath, threshold::Float64, dstpath)
    src = load(srcpath)
    tgt = load(tgtpath)

    src_mcep = src["feature_matrix"]
    tgt_mcep = tgt["feature_matrix"]

    # Perform alignment
    src_mcep, tgt_mcep = align_mcep(src_mcep, tgt_mcep,
                                    th=threshold,
                                    alpha=float(src["alpha"]),
                                    framelen=int(src["framelen"]))
    println("The number of aligned frames: $(size(src_mcep, 2))")
    if size(src_mcep, 2) ==  0
        @warn("No frame found in aligned data. Probably threshold is too high.")
    end

    @assert !any(isnan(src_mcep))
    @assert !any(isnan(tgt_mcep))

    src["feature_matrix"] = src_mcep
    tgt["feature_matrix"] = tgt_mcep

    # type Dict{Union(UTF8String, ASCIIString), Any} is saved as
    # Dict{UTF8String, Any} and cause error in reading JLD file.
    # remove off Union and then save do the trick (but why? bug in HDF5?)
    @assert isa(src, Dict{Union(UTF8String, ASCIIString), Any})
    @assert isa(tgt, Dict{Union(UTF8String, ASCIIString), Any})
    save(dstpath,
         "src", Dict{UTF8String,Any}(src),
         "tgt", Dict{UTF8String,Any}(tgt)
    )
end

function main()
    args = docopt(doc, version=v"0.0.1")

    srcdir = args["<src_dir>"]
    tgtdir = args["<tgt_dir>"]
    dstdir = args["<dst_dir>"]
    if !isdir(dstdir)
        @info("Create $(dstdir)")
        run(`mkdir -p $dstdir`)
    end

    const threshold = float(args["--threshold"])
    const nmax = int(args["--max"])

    files = searchdir(srcdir, ".jld")
    @info("$(length(files)) data found.")

    count = 0
    for filename in files
        # filename is assumed to be same between src and tgt
        srcpath = joinpath(srcdir, filename)
        tgtpath = joinpath(tgtdir, filename)
        dstpath = joinpath(dstdir, string(splitext(basename(srcpath))[1],
                                      "_parallel.jld"))

        @info("Start processing $(srcpath) and $(tgtpath)")
        elapsed = @elapsed _align(srcpath, tgtpath, threshold, dstpath)
        @info("Elapsed time in alignment is $(elapsed) sec.")
        @info("Dumped to $(dstpath)")

        count += 1
        if count >= nmax
            break
        end
    end

    @info("Finished")
end

main()
