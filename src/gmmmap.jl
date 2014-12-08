# GMM-based frame-by-frame voice conversion
#
# Reference:
# [Toda 2007] T. Toda, A. W. Black, and K. Tokuda, “Voice conversion based on
# maximum likelihood estimation of spectral parameter trajectory,” IEEE
# Trans. Audio, Speech, Lang. Process, vol. 15, no. 8, pp. 2222–2235, Nov. 2007.
# http://isw3.naist.jp/~tomoki/Tomoki/Journals/IEEE-Nov-2007_MLVC.pdf

# GMMMapParm represents a set of immutable parameters of GMM-based conversion
immutable GMMMapParam
    weights::Vector{Float64}
    μˣ::Matrix{Float64}
    μʸ::Matrix{Float64}
    Σˣˣ::Array{Float64, 3}
    Σˣʸ::Array{Float64, 3}
    Σʸˣ::Array{Float64, 3}
    Σʸʸ::Array{Float64, 3}

    # pre-computed in constructor to avoid dupulicate computation
    # in conversion process
    ΣʸˣΣˣˣ⁻¹::Array{Float64, 3}

    function GMMMapParam(weights::Vector{Float64},
                         μˣ::Matrix{Float64},
                         μʸ::Matrix{Float64},
                         Σˣˣ::Array{Float64, 3},
                         Σˣʸ::Array{Float64, 3},
                         Σʸˣ::Array{Float64, 3},
                         Σʸʸ::Array{Float64, 3})
        const M = length(weights)
        const D = size(μˣ, 1)
        # pre-allocation and pre-computations
        ΣʸˣΣˣˣ⁻¹ = Array(Float64, D, D, M)
        for m=1:M
            ΣʸˣΣˣˣ⁻¹[:,:,m] = Σʸˣ[:,:,m] * Σˣˣ[:,:,m]^-1
        end
        new(weights, μˣ, μʸ, Σˣˣ, Σˣʸ, Σʸˣ, Σʸʸ, ΣʸˣΣˣˣ⁻¹)
    end
end

# GMMMap represents a composite type to transform spectral features of a source
# speaker to that of a target speaker based on GMM of source and target joint
# spectral features.
type GMMMap <: FrameByFrameConverter
    params::GMMMapParam
    Eʸ::Matrix{Float64}    # Eq. (11)
    px::GMM

    function GMMMap(weights::Vector{Float64},# shape: (M,)
                    μ::Matrix{Float64},      # shape: (D, M)
                    Σ::Array{Float64,3};     # shape: (D, D, M)
                    swap::Bool=false)
        const M = length(weights)

        # Split mean and covariance matrices into source and target
        # speaker's ones
        const D = div(size(μ, 1), 2) # dimension of feature vector
        μˣ = μ[1:D, :]
        μʸ = μ[D+1:end, :]
        Σˣˣ = Σ[1:D,1:D, :]
        Σˣʸ = Σ[1:D,D+1:end, :]
        Σʸˣ = Σ[D+1:end,1:D, :]
        Σʸʸ = Σ[D+1:end,D+1:end, :]

        # swap src and target parameters
        if swap
            μˣ, μʸ = μʸ, μˣ
            Σˣˣ, Σʸʸ = Σʸʸ, Σˣˣ
            Σˣʸ, Σʸˣ = Σʸˣ, Σˣʸ
        end

        # construct params
        params = GMMMapParam(weights, μˣ, μʸ, Σˣˣ, Σˣʸ, Σʸˣ, Σʸʸ)

        ## pre-allocations
        Eʸ = zeros(D, M)

        # p(x)
        px = GaussianMixtureModel(μˣ, Σˣˣ, weights)

        new(params, Eʸ, px)
    end
end

Base.length(g::GMMMap) = 1 # GMMMap represents `frame-by-frame` converter
dim(g::GMMMap) = size(g.params.μˣ, 1)
ncomponents(g::GMMMap) = length(g.params.weights)
Base.size(g::GMMMap) = (dim(g), length(g))

# Mapping source spectral feature x to target spectral feature y
# so that minimize the mean least squared error.
# More specifically, it returns the value E(p(y|x)].
function fvconvert(g::GMMMap, x::Vector{Float64})
    dim(g) == length(x) || throw(DimensionMismatch("Inconsistent dimentions."))

    μˣ = g.params.μˣ
    μʸ = g.params.μʸ
    ΣʸˣΣˣˣ⁻¹ = g.params.ΣʸˣΣˣˣ⁻¹

    # Eq. (11)
    for m=1:ncomponents(g)
        @inbounds g.Eʸ[:,m] = μʸ[:,m] + (ΣʸˣΣˣˣ⁻¹[:,:,m]) * (x - μˣ[:,m])
    end

    # Eq. (9) p(m|x)
    posterior = predict_proba(g.px, x)

    # Eq. (13)
    g.Eʸ * posterior
end
