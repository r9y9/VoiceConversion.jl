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
    const D, T = div(size(X,1),2), size(X,2)
    # input feature vector must contain delta feature
    @assert D == tgmm.D*2
    
    if T != tgmm.T
        tgmm.W = construct_weight_matrix(D, T)
        tgmm.T = T
    end

    # A suboptimum mixture sequence  (eq.37)
    optimum_mix = predict(tgmm.gmmmap.px, X)
    
    # Compute E eq.(40)
    E = zeros(2*D, T)
    for t=1:T
        m = int(optimum_mix[t])
        E[:,t] = tgmm.gmmmap.tgt_means[:,m] + tgmm.gmmmap.covarYX_XXinv[:,:,m] * 
            (X[:,t] - tgmm.gmmmap.src_means[:,m])
    end

    # Compute D^-1 eq.(41)
    # TODO work
    Dinv = zeros(2*D, 2*D, T)
    for t=1:T
        m = int(optimum_mix[t])
        Dinv[:,:,t] = tgmm.gmmmap.covarYY[:,:,m] - tgmm.gmmmap.covarYX_XXinv[:,:,m] * 
            tgmm.gmmmap.covarXY[:,:,m]
        Dinv[:,:,t] = sparse(Dinv[:,:,t]^-1)
    end
    Dinv = blkdiag(Dinv)
    @assert size(Dinv) == (2*D*T, 2*D*T)
    @assert issparse(Dinv)

    # Compute target static features
    # eq.(39)
    y = (W' * Dinv * W)^-1 * W' * Dinv * E 
    @assert size(y) == (D, T)

    return y
end
