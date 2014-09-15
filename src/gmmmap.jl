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

    px::GMM{PDMat}

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
        px = GaussianMixtureModel(src_means, covarXX, weights)

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

# Trajectory-based speech parameter mapping for voice conversion
# based on the maximum likelihood criterion.
type TrajectoryGMMMap <: TrajectoryConverter
    gmmmap::GMMMap
    T::Int
    D::Int
    W::SparseMatrixCSC{Float64, Int}

    # TODO consider gloval variance (gv)
    function TrajectoryGMMMap(gmmmap::GMMMap, T::Int)
        const D = div(size(gmmmap.src_means, 1), 4)
        W = construct_weight_matrix(D, T)
        new(gmmmap, T, D, W)
    end
end

function construct_weight_matrix(D::Int, T::Int)
    W = spzeros(2*D*T, D*T)

    for t=1:T
        w0 = spzeros(D, D*T)
        w1 = spzeros(D, D*T)
        w0[1:end, (t-1)*D+1:t*D] = spdiagm(ones(D))
        
        if t-1 >= 1
            w1[1:end, (t-1)*D+1:t*D] = spdiagm(-0.5*ones(D))
        end
        if t < T
            w1[1:end, (t-1)*D+1:t*D] = spdiagm(0.5*ones(D))
        end
        
        W[2*D*(t-1)+1:2*D*t,:] = [w0, w1]
    end

    return W
end

function fvconvert(tgmm::TrajectoryGMMMap, X::Matrix{Float64})
    const D, T = size(X)
    # input feature vector must contain delta feature
    @assert D == tgmm.D*2
    
    if T != tgmm.T
        tgmm.W = construct_weight_matrix(D, T)
        tgmm.T = T
    end

    # A suboptimum mixture sequence  (eq.37)
    optimum_mix = predict(tgmm.gmmmap.px, src)
    
    # Compute E eq.(40)
    E = zeros(2*D, T)
    for t=1:T
        m = int(optimum_mix[t])
        E[:,t] = tgmm.gmmmap.tgt_means[:,m] + tgmm.gmmmap.covarYX_XXinv[:,:,m] * 
            (X[:,t] - tgmm.gmmmap.src_means[:,m])
    end

    # Compute D^-1 eq.(41)
    Dinv = zeros(2*D, 2*D, T)
    for t=1:T
        m = int(optimum_mix[t])
        Dinv[:,:,m] = tgmm.gmmmap.covarYY[:,:,m] - tgmm.gmmmap.covarYX_XXinv[:,:,m] * 
            tgmm.gmmmap.covarXY[:,:,m]
    end
    Dinv = sparse(Dinv)

    # Compute target static features
    # eq.(39)
    y = (W' * Dinv * W)^-1 * W' * Dinv * E 
    @assert size(y) == (D, T)

    return y
end
