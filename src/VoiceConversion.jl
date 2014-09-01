module VoiceConversion

# DTW related functions
export DTW, fit!, update!, set_template!, backward

# Feature conversion, extractions and alignment
export logamp2mcep, mcep2e, world_mcep, align_mcep

# Datasets
export Dataset, ParallelDataset

# Feature conversion
export FrameByFrameConverter, GMMMap, GMM, fvconvert, vc

include("align.jl")
include("dtw.jl")
include("datasets.jl")
include("feature.jl")
include("gmm.jl")
include("gmmmap.jl")
include("converter.jl")

end # module
