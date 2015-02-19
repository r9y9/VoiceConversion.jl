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
    --max=MAX           maximum number of data we use [default: 100]
    --min_covar=MIN     minimum covariance [default: 1.0e-7]
    --refine            refine pre-trained GMM (<dst_jld>)
"""

using VoiceConversion
using HDF5, JLD
using PyCall

# TODO: remove this dependency
@pyimport sklearn.mixture as mixture

using Logging
@Logging.configure(level=DEBUG, output=STDOUT)

# pygmmmean2jl return transposed parameters because julia is column-major
pygmmmean2jl(means::AbstractMatrix) = means'

# shape (M, D, D) -> (D, D, M)
# where M is the # of mixtures and D is the dimention of feature vector
function pygmmcovar2jl{T}(py_covars::Array{T,3})
    n_components = size(py_covars, 1)
    order = size(py_covars, 2)
    covars = Array(eltype(py_covars), order, order, n_components)
    for m=1:n_components
        covars[:,:,m] = reshape(py_covars[m,:,:], order, order)
    end
    covars
end

jlgmmmean2py(means::AbstractMatrix) = means'

# shape (D, D, M) -> (M, D, D)
function jlgmmcovar2py{T}(jl_covars::Array{T,3})
    n_components = size(jl_covars, 3)
    order = size(jl_covars, 2)
    covars = Array(eltype(jl_covars), n_components, order, order)
    for m=1:n_components
        covars[m,:,:] = reshape(jl_covars[:,:,m], 1, order, order)
    end
    covars
end

function copy2pygmm!(jlgmm, pygmm)
    pygmm[:means_] = jlgmmmean2py(jlgmm["means"])
    pygmm[:covars_] = jlgmmcovar2py(jlgmm["covars"])
    pygmm[:weights_] = jlgmm["weights"]
    pygmm[:init_params] = ""
end

let
    args = docopt(doc, version=v"0.0.2")
    diff = args["--diff"]
    n_components = int(args["--n_components"])
    n_iter = int(args["--n_iter"])
    n_init = int(args["--n_init"])
    min_covar = float64(args["--min_covar"])
    refine = args["--refine"]

    dataset = ParallelDataset(args["<parallel_dir>"],
                              joint=true,
                              diff=diff,
                              add_delta=args["--add_delta"],
                              nmax=int(args["--max"]))

    savepath = args["<dst_jld>"]

    gmm = mixture.GMM(n_components=n_components,
                      covariance_type="full",
                      n_iter=n_iter,
                      n_init=n_init,
                      min_covar=min_covar
                      )

    savepath = args["<dst_jld>"]
    if refine
        if !isfile(savepath)
            error("$(savepath) not found. Cannot refine model.")
        end
        println("refine pre-trained GMM")
        pretrained_jlgmm = load(savepath)
        copy2pygmm!(pretrained_jlgmm, gmm)
    end

    @show gmm
    # pass transposed matrix because python is row-major language
    elapsed = @elapsed gmm[:fit](dataset.X')
    @info("Elapsed time in training is $(elapsed) sec.")

    save(savepath,
         "description", "Parameters of Gaussian Mixture Model",
         "diff", diff,
         "n_components", length(gmm[:weights_]),
         "weights", gmm[:weights_],
         "means", pygmmmean2jl(gmm[:means_]),
         "covars", pygmmcovar2jl(gmm[:covars_])
         )

    @info("Dumped to $(savepath)")

    println("Finished")
end
