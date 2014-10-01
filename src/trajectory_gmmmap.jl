# Trajectory-based speech parameter mapping for voice conversion
# based on the maximum likelihood criterion.
type TrajectoryGMMMap <: TrajectoryConverter
    gmmmap::GMMMap
    T::Int
    D::Int
    W::SparseMatrixCSC{Float64, Int}

    # diagonal components of eq. (41)
    Dy::Array{Float64, 3}
    Dinv::SparseMatrixCSC{Float64, Int}
    # vectroized version of eq. (40)
    E::Vector{Float64}

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
        
        new(gmmmap, T, D, W, Dy, spzeros(0,0), zeros(0))
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
    tgmm.E = E # keep E for GV optimization
    @assert size(E) == (2*D*T,)
    @assert size(tgmm.E) == (2*D*T,)

    # Compute D^-1 eq.(41)
    Dinv = blkdiag([sparse(tgmm.Dy[:,:,optimum_mix[t]]) for t=1:T]...)
    tgmm.Dinv = Dinv # Keep Dinv for GV

    @assert size(Dinv) == (2*D*T, 2*D*T)
    @assert size(tgmm.Dinv) == (2*D*T, 2*D*T)
    @assert issparse(Dinv)
    @assert issparse(tgmm.Dinv)

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

type TrajectoryGMMMapWithGV <: TrajectoryConverter
    tgmm::TrajectoryGMMMap
    gv_mean::Vector{Float64}
    gv_covar::Matrix{Float64}
    pv::Matrix{Float64}

    function TrajectoryGMMMapWithGV(tgmm::TrajectoryGMMMap, 
                                    gv_mean, gv_covar)
        @assert sum(gv_mean .< 0) == 0
        new(tgmm, gv_mean, gv_covar, inv(gv_covar))
    end
    
    function TrajectoryGMMMapWithGV(tgmm::TrajectoryGMMMap,
                                    gvgmm::Dict{Union(UTF8String, ASCIIString)})
        # assume single mixture
        (size(gvgmm["means"], 2) == 1) || error("not supported for mixture >= 2")
        gv_mean = gvgmm["means"][:,1]
        @assert sum(gv_mean .< 0) == 0
        gv_covar = gvgmm["covars"][:,:,1]
        new(tgmm, gv_mean, gv_covar, inv(gv_covar))
    end
end

function fvconvert(tgv::TrajectoryGMMMapWithGV, X::Matrix{Float64};
                   epochs::Int=100, learning_rate::Float64=1.0e-5)
    # Initialize target static features without considering GV
    y = fvconvert(tgv.tgmm, X)
    const D, T = size(y)

    # eq. (58)
    y = sqrt(tgv.gv_mean ./ var(X[1:D,:], 2)) .* (y .- mean(y, 2)) .+ mean(y, 2)

    const ω = 1.0/(2.0*T)

    # aliases
    E = tgv.tgmm.E
    W = tgv.tgmm.W
    Dinv = tgv.tgmm.Dinv
    Wt_Dinv = W' * Dinv
    
    # Gradient decent
    for epoch=1:epochs
        grad_y = -Wt_Dinv * W * vec(y) + Wt_Dinv * E
        grad = ω*grad_y + vec(gvgrad(tgv, y))
        println("Epoch #$(epoch): norm $(norm(grad))")
        y = y + learning_rate * reshape(grad, D, T)
    end

    return y
end

# gvgrad computes gradient of the likelihood with regard to GV.
function gvgrad(tgv::TrajectoryGMMMapWithGV, y::Matrix{Float64})
    const D, T = size(y)
    
    gv = var(y, 2) # global variance over time
    @assert size(gv) == (D, 1)
    y_mean = mean(y, 2)
    @assert size(y_mean) == (D, 1)
    
    v = Array(Float64, D, T)
    for t=1:T
        @inbounds v[:,t] = -2.0/T*(tgv.pv'*(gv - tgv.gv_mean)) .* (y[:,t] - y_mean)
    end
    
    return v
end
