using Distributions

import NumericExtensions: logsumexp

# GMM represents Gaussian Mixture Models.
# TODO(ryuichi) should be more generic
type GMM
    n_components::Int
    normals::Vector{GenericMvNormal}
    weights::Vector{Float64}

    function GMM(means, covars, weights)
        const n_components::Int = size(means, 2)
        normals = Array(GenericMvNormal, n_components)
        for m=1:n_components
            normals[m] = MvNormal(means[:,m], covars[:,:,m])
        end
        new(n_components, normals, weights)
    end
end

ncomponents(gmm::GMM) = gmm.n_components

# predict_proba predicts posterior probability of data under eash Gaussian
# in the model.
# TODO(ryuichi) do for matrix x
function predict_proba(gmm::GMM, x::Vector{Float64})
    lpr = [logpdf(m, x)::Float64 for m in gmm.normals] + log(gmm.weights)
    logprob = logsumexp(lpr)
    posterior = exp(lpr - logprob)
end
