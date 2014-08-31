using DocOpt

doc="""Align two feature vector sequences (feature matrices) and create
time aligned joint vector sequence (paralell data)

Usage:
    align.jl [options] <src> <tgt> <dst>
    align.jl --version
    align.jl -h | --help

Options:
    -h --help       show this message
    --threshold=TH  threshold that is used to remove silence [default: 14.0]
"""

import VoiceConversion: align
using HDF5, JLD

function main()
    args = docopt(doc, version=v"0.0.1")

    src = load(args["<src>"])
    tgt = load(args["<tgt>"])

    src_mcep = src["feature_matrix"]
    tgt_mcep = tgt["feature_matrix"]

    # Perform alignment
    src_mcep, tgt_mcep = align(src_mcep, tgt_mcep,
                               th=float(args["--threshold"]),
                               alpha=float(src["alpha"]),
                               framelen=int(src["framelen"]))

    @assert !any(isnan(src_mcep))
    @assert !any(isnan(tgt_mcep))

    # save
    src["feature_matrix"] = src_mcep
    tgt["feature_matrix"] = tgt_mcep
    save(args["<dst>"], "src", src, "tgt", tgt)
end

main()
