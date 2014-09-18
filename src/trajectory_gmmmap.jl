# Trajectory-based speech parameter mapping for voice conversion
# based on the maximum likelihood criterion.
type TrajectoryGMMMap <: TrajectoryConverter
    gmmmap::GMMMap
    T::Int
    D::Int
    W::SparseMatrixCSC{Float64, Int}

    # diagonal components of eq. (41)
    Dy::Array{Float64, 3}

    # TODO consider gloval variance (gv)
    function TrajectoryGMMMap(gmmmap::GMMMap, T::Int)
        const D = div(size(gmmmap.src_means, 1), 2)
        W = construct_weight_matrix(D, T)

        # the number of mixtures
        const M = size(gmmmap.src_means, 2)

        # pre-computations
        Dy = Array(Float64, 2*D, 2*D, M)
        for m=1:M
            Dy[:,:,m] = gmmmap.covarYY[:,:,m] - gmmmap.covarYX_XXinv[:,:,m] *
                gmmmap.covarXY[:,:,m]
            Dy[:,:,m] = Dy[:,:,m]^-1
        end
        
        new(gmmmap, T, D, W, Dy)
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

    @assert issparse(W)
    @assert size(W) == (2*D*T, D*T)

    return W
end

function fvconvert(tgmm::TrajectoryGMMMap, X::Matrix{Float64})
    # input feature vector must contain delta feature
    const D, T = div(size(X,1),2), size(X,2)
    D == tgmm.D || throw(DimensionMismatch("Inconsistent dimentions."))
    
    if T != tgmm.T
        tgmm.W = construct_weight_matrix(D, T)
        tgmm.T = T
    end

    # A suboptimum mixture sequence  (eq.37)
    optimum_mix = predict(tgmm.gmmmap.px, X)
    
    # Compute E eq.(40)
    E = Array(Float64, 2*D, T)
    for t=1:T
        const m = int(optimum_mix[t])
        E[:,t] = tgmm.gmmmap.tgt_means[:,m] + tgmm.gmmmap.covarYX_XXinv[:,:,m] *
            (X[:,t] - tgmm.gmmmap.src_means[:,m])
    end
    E = vec(E)
    @assert size(E) == (2*D*T,)

    # Compute D^-1 eq.(41)
    Dinv = blkdiag([sparse(tgmm.Dy[:,:,optimum_mix[t]]) for t=1:T]...)

    @assert size(Dinv) == (2*D*T, 2*D*T)
    @assert issparse(Dinv)

    # Compute target static feature vector
    # eq.(39)
    W = tgmm.W # short alias
    Wt_Dinv = W' * Dinv
    @assert issparse(Wt_Dinv)
    y = (Wt_Dinv * W) \ (Wt_Dinv * E)
    @assert size(y) == (D*T,)

    # Finally we get static feature vector
    y = reshape(y, D, T)

    return y
end
