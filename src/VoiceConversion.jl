module VoiceConversion

export DTW, fit!, update!, set_template!, backward, logamp2mcep, mcep2e

export CMUArctic

include("dtw.jl")
include("feature.jl")
include("datasets.jl")

end # module
