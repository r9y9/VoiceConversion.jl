# Trajectory-based speech parameter mapping for voice conversion
# based on the maximum likelihood criterion.
type TrajectoryGMMMap <: TrajectoryConverter
    gmmmap::GMMMap
    T::Int
    D::Int
    W::SparseMatrixCSC{Float64, Int}

    # diagonal components of eq. (41)
    Dʸ::Array{Float64, 3}
    # Dʸ^-1 in eq. (41)
    Dʸ⁻¹::SparseMatrixCSC{Float64, Int}
    # vectroized version of eq. (40)
    Eʸ::Vector{Float64}

    function TrajectoryGMMMap(gmmmap::GMMMap, T::Int)
        # alias
        g = gmmmap
        
        const D = div(size(g.μˣ, 1), 2)
        W = construct_weight_matrix(D, T)

        # the number of mixtures
        const M = size(g.μˣ, 2)

        # pre-computations
        Dʸ = Array(Float64, 2D, 2D, M)
        for m=1:M
            Dʸ[:,:,m] = g.Σʸʸ[:,:,m] - g.ΣʸˣΣˣˣ⁻¹[:,:,m] * g.Σˣʸ[:,:,m]
            Dʸ[:,:,m] = Dʸ[:,:,m]^-1
        end
        
        new(g, T, D, W, Dʸ, spzeros(0,0), zeros(0))
    end
end

function construct_weight_matrix(D::Int, T::Int)
    W = spzeros(2D*T, D*T)

    for t=1:T
        w⁰ = spzeros(D, D*T)
        w¹ = spzeros(D, D*T)
        w⁰[1:end, (t-1)*D+1:t*D] = spdiagm(ones(D))
        
        if t-1 >= 1
            w¹[1:end, (t-1)*D+1:t*D] = spdiagm(-0.5*ones(D))
        end
        if t < T
            w¹[1:end, (t-1)*D+1:t*D] = spdiagm(0.5*ones(D))
        end
        
        W[2*D*(t-1)+1:2*D*t,:] = [w⁰, w¹]
    end

    @assert issparse(W)
    @assert size(W) == (2D*T, D*T)

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

    # alias
    g = tgmm.gmmmap

    # A suboptimum mixture sequence  eq. (37)
    m̂ = predict(g.px, X)
    
    # Compute Eʸ eq.(40)
    Eʸ = Array(Float64, 2*D, T)
    for t=1:T
        const m = int(m̂[t])
        Eʸ[:,t] = g.μʸ[:,m] + g.ΣʸˣΣˣˣ⁻¹[:,:,m] * (X[:,t] - g.μˣ[:,m])
    end
    Eʸ = vec(Eʸ)
    tgmm.Eʸ = Eʸ # keep Eʸ for GV optimization
    @assert size(Eʸ) == (2D*T,)
    @assert size(tgmm.Eʸ) == (2D*T,)

    # Compute D^-1 eq.(41)
    Dʸ⁻¹ = blkdiag([sparse(tgmm.Dʸ[:,:,m̂[t]]) for t=1:T]...)
    tgmm.Dʸ⁻¹ = Dʸ⁻¹ # Keep Dʸ⁻¹ for GV

    @assert size(Dʸ⁻¹) == (2D*T, 2D*T)
    @assert size(tgmm.Dʸ⁻¹) == (2D*T, 2D*T)
    @assert issparse(Dʸ⁻¹)
    @assert issparse(tgmm.Dʸ⁻¹)

    # Compute target static feature vector
    # eq. (39)
    W = tgmm.W # short alias
    WᵀDʸ⁻¹ = W' * Dʸ⁻¹
    @assert issparse(WᵀDʸ⁻¹)
    y = (WᵀDʸ⁻¹ * W) \ (WᵀDʸ⁻¹ * Eʸ)
    @assert size(y) == (D*T,)

    # Finally we get static feature vector
    return reshape(y, D, T)
end

type TrajectoryGMMMapWithGV <: TrajectoryConverter
    tgmm::TrajectoryGMMMap
    μᵛ::Vector{Float64}
    Σᵛᵛ::Matrix{Float64}
    pᵥ::Matrix{Float64}

    function TrajectoryGMMMapWithGV(tgmm::TrajectoryGMMMap, μᵛ, Σᵛᵛ)
        @assert sum(μᵛ .< 0) == 0
        new(tgmm, μᵛ, Σᵛᵛ, inv(Σᵛᵛ))
    end
    
    function TrajectoryGMMMapWithGV(tgmm::TrajectoryGMMMap,
                                    gvgmm::Dict{Union(UTF8String, ASCIIString)})
        # assume single mixture
        (size(gvgmm["means"], 2) == 1) || error("not supported for mixture >= 2")
        μᵛ = gvgmm["means"][:,1]
        @assert sum(μᵛ .< 0) == 0
        Σᵛᵛ = gvgmm["covars"][:,:,1]
        new(tgmm, μᵛ, Σᵛᵛ, inv(Σᵛᵛ))
    end
end

function fvconvert(tgv::TrajectoryGMMMapWithGV, X::Matrix{Float64};
                   epochs::Int=100, α::Float64=1.0e-5)
    # Initialize target static features without considering GV
    y⁰ = fvconvert(tgv.tgmm, X)
    const D, T = size(y⁰)

    # Better initial value based on eq. (58)
    y⁰ = sqrt(tgv.μᵛ ./ var(X[1:D,:], 2)) .* (y⁰ .- mean(y⁰, 2)) .+ mean(y⁰, 2)

    const ω = 1.0/(2T)

    # aliases
    Eʸ = tgv.tgmm.Eʸ
    W = tgv.tgmm.W
    Dʸ⁻¹ = tgv.tgmm.Dʸ⁻¹
    WᵀDʸ⁻¹ = W' * Dʸ⁻¹
    
    # update y based on gradient decent
    yⁱ = y⁰
    for epoch=1:epochs
        Δyⁱ = ω*(-WᵀDʸ⁻¹ * W * vec(yⁱ) + WᵀDʸ⁻¹ * Eʸ) + vec(gvgrad(tgv, yⁱ))
        println("Epoch #$(epoch): norm $(norm(Δyⁱ))")
        Δyⁱ = reshape(Δyⁱ, D, T)
        # eq. (52)
        yⁱ = yⁱ + α * Δyⁱ
    end

    return yⁱ
end

# gvgrad computes gradient of the likelihood with regard to GV.
function gvgrad(tgv::TrajectoryGMMMapWithGV, y::Matrix{Float64})
    const D, T = size(y)
    
    gv = var(y, 2) # global variance over time
    @assert size(gv) == (D, 1)
    μʸ = mean(y, 2)
    @assert size(μʸ) == (D, 1)
    
    v = Array(Float64, D, T)
    for t=1:T
        @inbounds v[:,t] = -2.0/T*(tgv.pᵥ'*(gv - tgv.μᵛ)) .* (y[:,t] - μʸ)
    end
    
    return v
end
