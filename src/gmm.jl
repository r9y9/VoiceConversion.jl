using Distributions
using PDMats
using ArrayViews

import NumericExtensions: logsumexp

typealias GMM{Cov<:AbstractPDMat} 
    MixtureModel{Multivariate,Continuous,GenericMvNormal{Cov}}

# proxy to MixtureModel{Multivariate,Continuous,GenericMvNormal{Cov}}
function GaussianMixtureModel(means, covars, weights)
    const n_components::Int = size(means, 2)
    normals = Array(MvNormal, n_components)
    for m=1:n_components
        normals[m] = MvNormal(means[:,m], covars[:,:,m])
    end
    MixtureModel(normals, weights)
end

# predict_proba predicts posterior probability of data under eash Gaussian
# in the model.
function predict_proba{Cov<:AbstractPDMat}(gmm::GMM{Cov}, x)
    lpr = [(logpdf(gmm.components[i],x)+log(gmm.probs[i]))::Float64
           for i in find(gmm.probs .> 0.)]
    logprob = logsumexp(lpr)
    posterior = exp(lpr - logprob)
end

function predict_proba!{Cov<:AbstractPDMat}(r::AbstractMatrix,
                                            gmm::GMM{Cov},
                                            X::DenseMatrix)
    for i in 1:size(X,2)
        @inbounds r[:,i] = predict_proba(gmm, view(X,:,i))
    end
    return r
end

function predict_proba{Cov<:AbstractPDMat}(gmm::GMM{Cov}, X::DenseMatrix)
    predict_proba!(Array(Float64, length(gmm.probs), size(X,2)), gmm, X)
end

# predict label for x.
function predict{Cov<:AbstractPDMat}(gmm::GMM{Cov}, x)
    posterior = predict_proba(gmm, x)
    indmax(posterior)
end

function predict!{Cov<:AbstractPDMat}(r::AbstractArray,
                                      gmm::GMM{Cov},
                                      X::DenseMatrix)
    for i in 1:size(X,2)
        @inbounds r[i] = predict(gmm, view(X,:,i))
    end
    return r
end

function predict{Cov<:AbstractPDMat}(gmm::GMM{Cov}, X::DenseMatrix)
    predict!(Array(Float64, size(X,2)), gmm, X)
end
