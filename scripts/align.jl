using DocOpt

doc="""Align two feature vector sequences and create parallel data in batch.
Note that filename of source and target feature is assumed to be same.

Usage:
    align.jl [options] <src_dir> <tgt_dir> <dst_dir>
    align.jl --version
    align.jl -h | --help

Options:
    -h --help       show this message
    --threshold=TH  threshold that is used to remove silence [default: 14.0]
    --max=MAX       Maximum number that will be processed [default: 100]
"""

using VoiceConversion
using HDF5, JLD

searchdir(path, key) = filter(x -> contains(x, key), readdir(path))

function _align(srcpath, tgtpath, threshold::Float64, dstpath)
    src = load(srcpath)
    tgt = load(tgtpath)

    src_mcep = src["feature_matrix"]
    tgt_mcep = tgt["feature_matrix"]

    # check version consistency
    v1 = src["jl-version"]
    v2 = src["jl-version"]
    if v1 != v2
        warn("$(filename) $(v1) != $(v2)
             different version of julia was used to create source and target feature jld")
    end
    if v1 != VERSION || v2 != VERSION
        warn("$(filename) $(v1) != $(VERSION) or $(v2) != $(VERSION)
             you are using different version of julia since jld data was created.")
    end

    # Perform alignment
    src_mcep, tgt_mcep = align_mcep(src_mcep, tgt_mcep,
                                    th=threshold,
                                    alpha=float(src["alpha"]),
                                    framelen=int(src["framelen"]))

    @assert !any(isnan(src_mcep))
    @assert !any(isnan(tgt_mcep))

    src["feature_matrix"] = src_mcep
    tgt["feature_matrix"] = tgt_mcep

    save(dstpath,
         "src", src,
         "tgt", tgt,
         "jl-version", VERSION)
end

function main()
    args = docopt(doc, version=v"0.0.1")

    srcdir = args["<src_dir>"]
    tgtdir = args["<tgt_dir>"]
    dstdir = args["<dst_dir>"]
    if !isdir(dstdir)
        info("Create $(dstdir)")
        mkdir(dstdir)
    end

    const threshold = float(args["--threshold"])
    const nmax = int(args["--max"])

    files = searchdir(srcdir, ".jld")
    info("$(length(files)) data found.")

    count = 0
    for filename in files
        # filename is assumed to be same between src and tgt
        srcpath = joinpath(srcdir, filename)
        tgtpath = joinpath(tgtdir, filename)
        dstpath = joinpath(dstdir, string(splitext(basename(srcpath))[1],
                                      "_parallel.jld"))

        info("Start processing $(srcpath) and $(tgtpath)")
        elapsed = @elapsed _align(srcpath, tgtpath, threshold, dstpath)
        info("Elapsed time $(elapsed)")
        info("Dumped to $(dstpath)")

        count += 1
        if count >= nmax
            break
        end
    end

    println("Finished")
end

main()
