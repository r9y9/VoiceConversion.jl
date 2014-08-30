module VoiceConversion

export DTW, fit!, update!, set_template!, backward, logamp2mcep, mcep2e,
       world_mcep, align

export CMUArctic

include("align.jl")
include("dtw.jl")
include("feature.jl")
include("datasets.jl")

end # module
