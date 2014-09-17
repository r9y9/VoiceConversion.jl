# Trajectory-based speech parameter mapping for voice conversion
# based on the maximum likelihood criterion.
type TrajectoryGMMMap <: TrajectoryConverter
    gmmmap::GMMMap
    T::Int
    D::Int
    W::SparseMatrixCSC{Float64, Int}

    # TODO consider gloval variance (gv)
    function TrajectoryGMMMap(gmmmap::GMMMap, T::Int)
        const D = div(size(gmmmap.src_means, 1), 2)
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
    E = zeros(2*D, T)
    for t=1:T
        const m = int(optimum_mix[t])
        E[:,t] = tgmm.gmmmap.tgt_means[:,m] + tgmm.gmmmap.covarYX_XXinv[:,:,m] * 
            (X[:,t] - tgmm.gmmmap.src_means[:,m])
    end
    E = vec(E)
    @assert size(E) == (2*D*T,)

    # Compute D^-1 eq.(41)
    Dinv = spzeros(2*D, 2*D)
    for t=1:T
        const m = int(optimum_mix[t])
        dinv = tgmm.gmmmap.covarYY[:,:,m] - tgmm.gmmmap.covarYX_XXinv[:,:,m] * 
            tgmm.gmmmap.covarXY[:,:,m]
        dinv = dinv^-1
        if t == 1
            Dinv[:,:] = sparse(dinv)
        else
            Dinv = blkdiag(Dinv, sparse(dinv))
        end
    end
    # TODO: how can i pass diags as varargs to blkdiag..?
    # Dinv = blkdiag([Dinv[:,:,t] for t=1:T])

    @assert size(Dinv) == (2*D*T, 2*D*T)
    @assert issparse(Dinv)

    # Compute target static feature vector
    # eq.(39)
    W = tgmm.W # short alias
    Wt_Dinv = W' * Dinv
    @assert issparse(Wt_Dinv)
    y = full(Wt_Dinv * W)^-1 * Wt_Dinv * E
    @assert size(y) == (D*T,)

    # Finally we get static feature vector
    y = reshape(y, D, T)

    return y
end
