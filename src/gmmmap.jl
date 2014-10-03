# GMM-based frame-by-frame voice conversion
#
# Reference:
# [Toda 2007] T. Toda et al, “Voice conversion based on maximum likelihood
# estimation of spectral parameter trajectory,” IEEE
# Trans. Audio, Speech, Lang. Process, vol. 15, no. 8, pp. 2222–2235,
# Nov. 2007.
# http://isw3.naist.jp/~tomoki/Tomoki/Journals/IEEE-Nov-2007_MLVC.pdf

# GMMMap represents a class to transform spectral features of a source
# speaker to that of a target speaker based on Gaussian Mixture Models
# of source and target joint spectral features.
type GMMMap <: FrameByFrameConverter
    n_components::Int
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

    # Eq. (12) in [Toda2007]
    Dʸ::Array{Float64, 3}
    Eʸ::Matrix{Float64}

    px::GMM{PDMat}

    function GMMMap(gmm::Dict{Union(UTF8String, ASCIIString), Any};
                    swap::Bool=false)
        const n_components = gmm["n_components"]
        weights = gmm["weights"]
        @assert n_components == length(weights)
        μ = gmm["means"]
        Σ = gmm["covars"]

        # Split mean and covariance matrices into source and target
        # speaker's ones
        const order::Int = int(size(μ, 1) / 2)
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

        # pre-allocation and pre-computations
        ΣʸˣΣˣˣ⁻¹ = Array(Float64, order, order, n_components)
        for m=1:n_components
            ΣʸˣΣˣˣ⁻¹[:,:,m] = Σʸˣ[:,:,m] * Σˣˣ[:,:,m]^-1
        end       

        # Eq. (12)
        # Construct covariance matrices of p(Y|X) (Eq. (10))
        Dʸ = Array(Float64, order, order, n_components)
        for m=1:n_components
            Dʸ[:,:,m] = Σʸʸ[:,:,m] - Σʸˣ[:,:,m] * Σˣˣ[:,:,m]^-1 * Σˣʸ[:,:,m]
        end

        # pre-allocation
        Eʸ = zeros(order, n_components)

        # p(x)
        px = GaussianMixtureModel(μˣ, Σˣˣ, weights)

        new(n_components, weights, μˣ, μʸ,
            Σˣˣ, Σˣʸ, Σʸˣ, Σʸʸ, ΣʸˣΣˣˣ⁻¹, Dʸ, Eʸ, px)
    end
end

# Mapping source spectral feature x to target spectral feature y
# so that minimize the mean least squared error.
# More specifically, it returns the value E(p(y|x)].
function fvconvert(gmm::GMMMap, x::Vector{Float64})
    const order = length(x)

    # Eq. (11)
    for m=1:gmm.n_components
        @inbounds begin
            gmm.Eʸ[:,m] = gmm.μʸ[:,m] +
                (gmm.ΣʸˣΣˣˣ⁻¹[:,:,m]) * (x - gmm.μˣ[:,m])
        end
    end

    # Eq. (9) p(m|x)
    posterior = predict_proba(gmm.px, x)

    # Eq. (13)
    return gmm.Eʸ * posterior
end
