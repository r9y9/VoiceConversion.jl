# Gaussian Mixture Model (GMM)

import Compat: view

"""
Alias to MixtureModel{Multivariate,Continuous,MvNormal{Cov, Mean}}
"""
@compat const GaussianMixtureModel{Cov,Mean} = MixtureModel{Multivariate,Continuous,MvNormal{Cov,Mean}}

function (::Type{GaussianMixtureModel})(means, covars, weights;
    forcepd::Bool=true)
    n_components = size(means, 2)
    normals = Array(MvNormal, n_components)
    for m=1:n_components
        # force covariance matrix to be positive definite to avoid error on
        # creating PDMat internally
        # TODO: is this really correct?
        # caused by https://github.com/JuliaLang/julia/pull/16799
        covar = forcepd ? Array(Hermitian(covars[:,:,m])) : covars[:,:,m]
        normals[m] = MvNormal(means[:,m], covar)
    end
    MixtureModel(normals, weights)
end

# predict_proba predicts posterior probability of data under each Gaussian
# in the model.
function predict_proba(gmm::GaussianMixtureModel, x)
    p = probs(gmm)
    lpr = [(logpdf(gmm.components[i],x)+log(p[i]))::Float64
           for i in find(p .> 0.)]
    logprob = logsumexp(lpr)
    posterior = exp.(lpr - logprob)
end

function predict_proba!(r::AbstractMatrix, gmm::GaussianMixtureModel, X::DenseMatrix)
    for i in 1:size(X,2)
        @inbounds r[:,i] = predict_proba(gmm, view(X,:,i))
    end
    r
end

function predict_proba(gmm::GaussianMixtureModel, X::DenseMatrix)
    predict_proba!(Array(Float64, length(probs(gmm)), size(X,2)), gmm, X)
end

# predict label for x.
function predict(gmm::GaussianMixtureModel, x)
    posterior = predict_proba(gmm, x)
    indmax(posterior)::Int
end

function predict!(r::AbstractArray, gmm::GaussianMixtureModel, X::DenseMatrix)
    for i in 1:size(X,2)
        @inbounds r[i] = predict(gmm, view(X,:,i))
    end
    r
end

function predict(gmm::GaussianMixtureModel, X::DenseMatrix)
    predict!(Array(Int, size(X,2)), gmm, X)
end
