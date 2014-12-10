# Gaussian Mixture Model (GMM)

typealias GMM{Cov,Mean} MixtureModel{Multivariate,Continuous,MvNormal{Cov,Mean}}

# proxy to MixtureModel{Multivariate,Continuous,MvNormal{Cov, Mean}}
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
function predict_proba(gmm::GMM, x)
    p = probs(gmm)
    lpr = [(logpdf(gmm.components[i],x)+log(p[i]))::Float64
           for i in find(p .> 0.)]
    logprob = logsumexp(lpr)
    posterior = exp(lpr - logprob)
end

function predict_proba!(r::AbstractMatrix, gmm::GMM, X::DenseMatrix)
    for i in 1:size(X,2)
        @inbounds r[:,i] = predict_proba(gmm, view(X,:,i))
    end
    r
end

function predict_proba(gmm::GMM, X::DenseMatrix)
    predict_proba!(Array(Float64, length(probs(gmm)), size(X,2)), gmm, X)
end

# predict label for x.
function predict(gmm::GMM, x)
    posterior = predict_proba(gmm, x)
    indmax(posterior)
end

function predict!(r::AbstractArray, gmm::GMM, X::DenseMatrix)
    for i in 1:size(X,2)
        @inbounds r[i] = predict(gmm, view(X,:,i))
    end
    r
end

function predict(gmm::GMM, X::DenseMatrix)
    predict!(Array(Float64, size(X,2)), gmm, X)
end
