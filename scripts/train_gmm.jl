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
    --refine            refine pre-trained GMM (<dst_jld>)
"""

using VoiceConversion
using HDF5, JLD
using PyCall

@pyimport sklearn.mixture as mixture

# pygmmmean2jl return transposed parameters because julia is column-major
pygmmmean2jl(means::Matrix{Float64}) = means'

# shape (M, D, D) -> (D, D, M)
# where M is the # of mixtures and D is the dimention of feature vector
function pygmmcovar2jl(py_covars::Array{Float64,3})
    const n_components = size(py_covars, 1)
    const order = size(py_covars, 2)
    covars = Array(eltype(py_covars), order, order, n_components)
    for m=1:n_components
        covars[:,:,m] = reshape(py_covars[m,:,:], order, order)
    end
    covars
end

jlgmmmean2py(means::Matrix{Float64}) = means'

# shape (D, D, M) -> (M, D, D)
function jlgmmcovar2py(jl_covars::Array{Float64,3})
    const n_components = size(jl_covars, 3)
    const order = size(jl_covars, 2)
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

function main()
    args = docopt(doc, version=v"0.0.2")

    const nmax::Int = int(args["--max"])
    const diff = args["--diff"]
    const add_delta = args["--add_delta"]

    const n_components::Int = int(args["--n_components"])
    const n_iter::Int = int(args["--n_iter"])
    const n_init::Int = int(args["--n_init"])
    const min_covar::Float64 = float64(args["--min_covar"])
    const refine = args["--refine"]

    dataset = ParallelDataset(args["<parallel_dir>"],
                              joint=true,
                              diff=diff,
                              add_delta=add_delta,
                              nmax=nmax)

    gmm = mixture.GMM(n_components=n_components,
                      covariance_type="full",
                      n_iter=n_iter,
                      n_init=n_init,
                      min_covar=min_covar
                      )

    dstpath = args["<dst_jld>"]
    if refine
        if !isfile(dstpath)
            error("$(dstpath) not found. Cannot refine model.")
        end
        println("refine pre-trained GMM")
        pretrained_jlgmm = load(dstpath)
        copy2pygmm!(pretrained_jlgmm, gmm)
    end

    @show gmm

    X = dataset.X

    # pass transposed matrix because python is row-major language
    elapsed = @elapsed gmm[:fit](X')
    info("Elapsed time in training: $(elapsed)")

    save(dstpath,
         "description", "Parameters of Gaussian Mixture Model",
         "diff", diff,
         "n_components", n_components,
         "weights", gmm[:weights_],
         "means", pygmmmean2jl(gmm[:means_]),
         "covars", pygmmcovar2jl(gmm[:covars_]),
         "jl-version", VERSION
         )

    info("Dumped to $(dstpath)")
end

main()
