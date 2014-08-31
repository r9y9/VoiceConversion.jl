using DocOpt

doc="""Align two feature vector sequences and create parallel data in batch.

Usage:
    align_batch.jl [options] <src_dir> <tgt_dir> <dst_dir>
    align_batch.jl --version
    align_batch.jl -h | --help

Options:
    -h --help       show this message
    --threshold=TH  threshold that is used to remove silence [default: 14.0]
    --max=MAX       Maximum number that will be processed [default: 100]
"""

using VoiceConversion
using HDF5, JLD

searchdir(path, key) = filter(x -> contains(x, key), readdir(path))

function main()
    args = docopt(doc, version=v"0.0.1")

    srcdir = args["<src_dir>"]
    tgtdir = args["<tgt_dir>"]
    dstdir = args["<dst_dir>"]
    if !isdir(dstdir)
        info("Create $(dstdir)")
        mkdir(dstdir)
    end

    const nmax = int(args["--max"])

    files = searchdir(srcdir, ".jld")
    info("$(length(files)) data found.")

    count = 0
    for filename in files
        srcpath = joinpath(srcdir, filename)
        # filename is assumed to be same 
        tgtpath = joinpath(tgtdir, filename)

        src = load(srcpath)
        tgt = load(tgtpath)

        src_mcep = src["feature_matrix"]
        tgt_mcep = tgt["feature_matrix"]

        # Perform alignment
        src_mcep, tgt_mcep = align(src_mcep, tgt_mcep,
                                   th=float(args["--threshold"]),
                                   alpha=float(src["alpha"]),
                                   framelen=int(src["framelen"]))

        src["feature_matrix"] = src_mcep
        tgt["feature_matrix"] = tgt_mcep
        dstpath = joinpath(dstdir, string(splitext(basename(srcpath))[1],
                                          "_parallel.jld"))
        save(dstpath, "src", src, "tgt", tgt)
        
        info("Dumped to $(dstpath)")        

        count += 1
        if count >= nmax
            break
        end
    end
    
    println("Finished")
end

@time main()
