@pyimport sklearn.mixture as mixture

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

function _train_gmm(X::AbstractMatrix;
                    n_components=16,
                    n_iter=200,
                    n_init=2,
                    min_covar=1e-7,
                    covariance_type="full",
                    refine::Bool=false,
                    pretrained_gmm_path=nothing
    )
    covariance_type != "full" && error("covariance type full is only supported")

    gmm = mixture.GMM(n_components=n_components,
                      covariance_type=covariance_type,
                      n_iter=n_iter,
                      n_init=n_init,
                      min_covar=min_covar
                      )

    if refine
        if !isfile(pretrained_gmm_path)
            error("$(dstpath) not found. Cannot refine model.")
        end
        @info("refine pre-trained GMM")
        pretrained_jlgmm = load(pretrained_gmm_path)
        copy2pygmm!(pretrained_jlgmm, gmm)
    end

    @info("showing GMM configurations")
    @show gmm

    # pass transposed matrix because python is row-major language
    @info("Start training GMM using sklearn.mixture.")
    elapsed = @elapsed gmm[:fit](X')
    @info("Elapsed time in training $(elapsed/3600) hours.")

    gmm
end

function save_gmm(savepath, gmm;
                  diff=nothing)

    if diff != nothing
        save(savepath,
             "description", "Parameters of Gaussian Mixture Model",
             "diff", diff,
             "n_components", length(gmm[:weights_]),
             "weights", gmm[:weights_],
             "means", pygmmmean2jl(gmm[:means_]),
             "covars", pygmmcovar2jl(gmm[:covars_])
             )
    else
        save(savepath,
             "description", "Parameters of Gaussian Mixture Model",
             "n_components", length(gmm[:weights_]),
             "weights", gmm[:weights_],
             "means", pygmmmean2jl(gmm[:means_]),
             "covars", pygmmcovar2jl(gmm[:covars_])
             )
    end

    @info("Dumped to $(savepath)")
end

hasattr(x, attr::Symbol) = length(filter(x -> x == attr, names(x))) > 0

function train_gmm(dataset::Dataset,
                   savepath;
                   n_components::Integer=16,
                   n_iter::Integer=200,
                   n_init::Integer=2,
                   min_covar::FloatingPoint=1e-7,
                   covariance_type="full",
                   refine::Bool=false,
                   pretrained_gmm_path=nothing
                   )
    gmm = _train_gmm(dataset.X,
                     n_components=n_components,
                     n_iter=n_iter,
                     n_init=2,
                     min_covar=1.0e-7,
                     covariance_type="full",
                     refine=refine,
                     pretrained_gmm_path=pretrained_gmm_path)
    gmm
end
