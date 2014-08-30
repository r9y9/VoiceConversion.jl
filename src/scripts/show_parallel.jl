using DocOpt

doc="""Visualize parallel data

Usage:
    show_parallel.jl [options] <parallel_jld>
    show_parallel.jl --version
    show_parallel.jl -h | --help

Options:
    -h --help         show this message
    -o --order=ORDER  order to be visualized [default: 1]
"""

using VoiceConversion
using HDF5, JLD
using PyCall

@pyimport matplotlib.pyplot as plt

function show_parallel(src, tgt; order=1)
    src_mcep = src["feature_matrix"]
    tgt_mcep = tgt["feature_matrix"]

    @assert size(src_mcep) == size(tgt_mcep)
    @assert order <= size(src_mcep, 1)

    plt.plot(src_mcep[order,:][:], color="red", linewidth=1.5)
    plt.plot(tgt_mcep[order,:][:], color="blue", linewidth=1.5)
    plt.xlabel("frame #")
    plt.title("Time aligned mel-cesptrum ($(order)-th order)")
    plt.show()

    nothing
end

function main()
    args = docopt(doc, version=v"0.0.1")
    order = int(args["--order"])

    parallel_data = load(args["<parallel_jld>"])

    src, tgt = parallel_data["src"], parallel_data["tgt"]

    show_parallel(src, tgt, order=order)

    nothing
end

@time main()
