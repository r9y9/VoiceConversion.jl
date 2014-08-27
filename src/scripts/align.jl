using DocOpt

doc="""Align two mel-cesptrum sequences and create parallel data for training
voice conversion models.
b
Usage:
    align.jl [options] <src_mcep> <tgt_mcep> <dst>
    align.jl --version
    align.jl -h | --help

Options:
    -h --help  show this message
"""

using VoiceConversion
using HDF5, JLD
using PyPlot

function main()
    args = docopt(doc, version=v"0.0.1")

    src = load(args["<src_mcep>"])
    tgt = load(args["<tgt_mcep>"])

    # for debug
    dump(src)
    dump(tgt)

    src_mcep = src["mcgram"]
    tgt_mcep = tgt["mcgram"]

    # Alignment
    d = DTW(fstep=0, bstep=2) # allow one skip
    path = fit!(d, src_mcep, tgt_mcep)

    # create aligned tgt_mcep
    newtgt_mcep = zeros(eltype(src_mcep), size(src_mcep))
    newtgt_mcep[:,path] = tgt_mcep[:,1:length(path)]

    # interpolation
    # TODO(ryuichi) better solution
    hole = setdiff([path[1]:path[end]], path)
    for i=hole[1]:hole[end]
        if i > 1 && i < size(src_mcep, 2)
            newtgt_mcep[:,i] =
                (newtgt_mcep[:,i-1] + newtgt_mcep[:,i+1]) / 2.0
        end
    end

    # remove silence segment
    # TODO(ryuichi)


    # plot (to be removed)
    plot(src_mcep[1,:][:])
    plot(newtgt_mcep[1,:][:])
    sleep(3.0)
    show()

    # save
    tgt["mcgram"] = newtgt_mcep
    save(args["<dst>"], "src", src, "tgt", tgt)
end

@time main()
