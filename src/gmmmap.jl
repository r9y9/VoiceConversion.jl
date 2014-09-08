# GMMMap represents a class to transform spectral features of a source
# speaker to that of a target speaker based on Gaussian Mixture Models
# of source and target joint spectral features.
type GMMMap <: FrameByFrameConverter
    n_components::Int
    weights::Vector{Float64}
    src_means::Matrix{Float64}
    tgt_means::Matrix{Float64}
    covarXX::Array{Float64, 3}
    covarXY::Array{Float64, 3}
    covarYX::Array{Float64, 3}
    covarYY::Array{Float64, 3}

    # pre-computed in constructor to avoid dupulicate computation
    # in conversion process
    covarYX_XXinv::Array{Float64, 3}

    # Eq. (12) in [Toda2007]
    D::Array{Float64, 3}
    E::Matrix{Float64}

    px::GMM

    function GMMMap(gmm::Dict{Union(UTF8String, ASCIIString), Any};
                    swap::Bool=false)
        const n_components = gmm["n_components"]
        weights = gmm["weights"]
        @assert n_components == length(weights)
        means = gmm["means"]
        covars = gmm["covars"]

        # Split mean and covariance matrices into source and target
        # speaker's ones
        const order::Int = int(size(means, 1) / 2)
        src_means = means[1:order, :]
        tgt_means = means[order+1:end, :]
        covarXX = covars[1:order,1:order, :]
        covarXY = covars[1:order,order+1:end, :]
        covarYX = covars[order+1:end,1:order, :]
        covarYY = covars[order+1:end,order+1:end, :]

        # swap src and target parameters
        if swap
            src_means, tgt_means = tgt_means, src_means
            covarXX, covarYY = covarYY, covarXX
            covarXY, covarYX = covarYX, covarXY
        end

        # pre-allocation and pre-computations
        covarYX_XXinv = zeros(order, order, n_components)
        for m=1:n_components
            covarYX_XXinv[:,:,m] = covarYX[:,:,m] * covarXX[:,:,m]^-1
        end       

        # Eq. (12)
        # Construct covariance matrices of p(Y|X) (Eq. (10))
        D = zeros(order, order, n_components)
        for m=1:n_components
            D[:,:,m] = covarYY[:,:,m] - covarYX[:,:,m] *
                covarXX[:,:,m]^-1 * covarXY[:,:,m]
        end

        # pre-allocation
        E = zeros(order, n_components)

        # p(x)
        px = GMM(src_means, covarXX, weights)

        new(n_components, weights, src_means, tgt_means,
            covarXX, covarXY, covarYX, covarYY, covarYX_XXinv, D, E, px)
    end
end

# Mapping source spectral feature x to target spectral feature y
# so that minimize the mean least squared error.
# More specifically, it returns the value E(p(y|x)].
function fvconvert(gmm::GMMMap, x::Vector{Float64})
    const order = length(x)

    # Eq. (11)
    E = gmm.tgt_means
    @inbounds for m=1:gmm.n_components
        gmm.E[:,m] += gmm.covarYX_XXinv[:,:,m] * (x - gmm.src_means[:,m])
    end

    # Eq. (9) p(m|x)
    posterior = predict_proba(gmm.px, x)

    # Eq. (13)
    return E * posterior
end

type TrajectoryGMMMap <: TrajectoryConverter
    # TODO
end

function fvconvert(tgmm::TrajectoryGMMMap, x::Matrix{Float64})
    # TODO
end
