# Reference:
# [Kobayashi 2014] Kobayashi, Kazuhiro, et al. "Statistical Singing Voice 
# Conversion with Direct Waveform Modification based on the Spectrum
# Differential." Fifteenth Annual Conference of the International Speech 
# Communication Association. 2014.

# DIFFGMM construction based on [Kobayashi 2014]
# TODO: tests
function diffgmm(params::GMMMapParam)
    μˣ = params.μˣ
    μʸ = params.μʸ
    Σˣˣ = params.Σˣˣ
    Σˣʸ = params.Σˣʸ
    Σʸˣ = params.Σʸˣ
    Σʸʸ = params.Σʸʸ

    GMMMapParam(params.weights, 
                μˣ,
                μʸ-μˣ, # Eq. (6) in [Kobayashi 2014]
                Σˣˣ,
                Σˣʸ - Σˣˣ, # Eq. (7)
                (Σˣʸ - Σˣˣ)',
                Σˣˣ + Σʸʸ - Σˣʸ - Σʸˣ # Eq. (8)
                )
end
