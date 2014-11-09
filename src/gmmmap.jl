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
        const order = size(μˣ, 1)
        # pre-allocation and pre-computations
        ΣʸˣΣˣˣ⁻¹ = Array(Float64, order, order, M)
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

    function GMMMap(gmm::Dict{Union(UTF8String, ASCIIString), Any};
                    swap::Bool=false)
        weights = gmm["weights"]
        const M = length(weights)
        μ = gmm["means"]
        Σ = gmm["covars"]

        # Split mean and covariance matrices into source and target
        # speaker's ones
        const order = div(size(μ, 1), 2)
        μˣ = μ[1:order, :]
        μʸ = μ[order+1:end, :]
        Σˣˣ = Σ[1:order,1:order, :]
        Σˣʸ = Σ[1:order,order+1:end, :]
        Σʸˣ = Σ[order+1:end,1:order, :]
        Σʸʸ = Σ[order+1:end,order+1:end, :]

        # swap src and target parameters
        if swap
            μˣ, μʸ = μʸ, μˣ
            Σˣˣ, Σʸʸ = Σʸʸ, Σˣˣ
            Σˣʸ, Σʸˣ = Σʸˣ, Σˣʸ
        end

        # construct params
        params = GMMMapParam(weights, μˣ, μʸ, Σˣˣ, Σˣʸ, Σʸˣ, Σʸʸ)

        ## pre-allocations
        Eʸ = zeros(order, M)

        # p(x)
        px = GaussianMixtureModel(μˣ, Σˣˣ, weights)

        new(params, Eʸ, px)
    end
end

ncomponents(g::GMMMap) = length(g.params.weights)

# Mapping source spectral feature x to target spectral feature y
# so that minimize the mean least squared error.
# More specifically, it returns the value E(p(y|x)].
function fvconvert(g::GMMMap, x::Vector{Float64})
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
