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
        means = gmm["means"]
        covars = gmm["covars"]

        # Split mean and covariance matrices into source and target
        # speaker's ones
        const order::Int = int(size(means, 1) / 2)
        μˣ = means[1:order, :]
        μʸ = means[order+1:end, :]
        Σˣˣ = covars[1:order,1:order, :]
        Σˣʸ = covars[1:order,order+1:end, :]
        Σʸˣ = covars[order+1:end,1:order, :]
        Σʸʸ = covars[order+1:end,order+1:end, :]

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
    Eʸ = gmm.μʸ
    @inbounds for m=1:gmm.n_components
        gmm.Eʸ[:,m] += (gmm.ΣʸˣΣˣˣ⁻¹[:,:,m]) * (x - gmm.μˣ[:,m])
    end

    # Eq. (9) p(m|x)
    posterior = predict_proba(gmm.px, x)

    # Eq. (13)
    return Eʸ * posterior
end
