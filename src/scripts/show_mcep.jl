using DocOpt

doc="""Visualize mel-cesptrum from JLD

Usage:
    show_mcep.jl [options] <mcep_jld>
    show_mcep.jl --version
    show_mcep.jl -h | --help

Options:
    -h --help         show this message
"""

using VoiceConversion
using HDF5, JLD
using PyCall

@pyimport matplotlib.pyplot as plt

function show_mcep2d(src)
    src_mcep = src["feature_matrix"]

    plt.figure(figsize=(12, 6), dpi=80, facecolor="w", edgecolor="k")
    plt.imshow(src_mcep, origin="lower", aspect="auto",
               interpolation="nearest")
    plt.colorbar()
    plt.show()

    nothing
end

function main()
    args = docopt(doc, version=v"0.0.1")
    mcep_jld = load(args["<mcep_jld>"])

    show_mcep2d(mcep_jld)
end

@time main()
