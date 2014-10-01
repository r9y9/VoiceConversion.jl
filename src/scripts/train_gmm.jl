using DocOpt

doc="""Training Gaussian Mixture models for Voice Conversion.

Usage:
    train_gmm.jl [options] <parallel_dir> <dst_jld>
    train_gmm.jl --version
    train_gmm.jl -h | --help

Options:
    -h --help           show this message
    --diff              use differencial
    --add_delta         add delta feature
    --n_components=MIX  number of mixtures [default: 16]
    --n_iter=ITER       number of iterations [default: 200]
    --n_init=N          number of initialization [default: 2]
    --max=MAX           maximum number of data we use [default: 50]
    --min_covar=MIN     minimum covariance [default: 1.0e-7]
    --src_only          use only source feature
    --tgt_only          use only target feature
"""

using VoiceConversion
using HDF5, JLD
using PyCall

@pyimport sklearn.mixture as mixture

function main()
    args = docopt(doc, version=v"0.0.1")

    const nmax::Int = int(args["--max"])
    const diff = args["--diff"]
    const add_delta = args["--add_delta"]

    const n_components::Int = int(args["--n_components"])
    const n_iter::Int = int(args["--n_iter"])
    const n_init::Int = int(args["--n_init"])
    const min_covar::Float64 = float64(args["--min_covar"])
    const src_only = args["--src_only"]
    const tgt_only = args["--tgt_only"]
    const joint = !src_only && !tgt_only
    @show joint

    (src_only && tgt_only) && error("invalid option")

    dataset = ParallelDataset(args["<parallel_dir>"],
                              joint=joint,
                              diff=diff,
                              add_delta=add_delta,
                              nmax=nmax)

    gmm = mixture.GMM(n_components=n_components,
                      covariance_type="full",
                      n_iter=n_iter,
                      n_init=n_init,
                      min_covar=min_covar
                      )

    @show gmm

    X = dataset.X
    if tgt_only
        X = dataset.Y
    end

    # pass transposed matrix because python is row-major language
    @time gmm[:fit](X')

    # save transposed parameters because julia is column-major language
    # convert means
    py_means = gmm[:means_]
    means = py_means'

    # convert covar tensor
    py_covars = gmm[:covars_] # shape: (n_components, order, order)
    order = size(py_covars, 2)
    covars = Array(eltype(py_covars), order, order, n_components)
    for m=1:n_components
        covars[:,:,m] = reshape(py_covars[m,:,:], order, order)
    end

    save(args["<dst_jld>"],
         "description", "Parameters of Gaussian Mixture Model",
         "diff", diff,
         "n_components", n_components,
         "weights", gmm[:weights_],
         "means", means,
         "covars", covars
         )
end

@time main()
