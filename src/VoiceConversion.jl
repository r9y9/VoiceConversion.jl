module VoiceConversion

# Dynamic Time Warping (DTW) related functions
export DTW, fit!, update!, set_template!, backward

# Feature conversion, extractions and alignment
export logamp2mcep, mcep2e, world_mcep, align_mcep

# Datasets
export ParallelDataset, push_delta

# Feature conversion
export Converter, FrameByFrameConverter, TrajectoryConverter,
       GMMMap, GMM, GaussianMixtureModel, fvconvert, vc,
       TrajectoryGMMMap

## Type Hierarchy ##
abstract Converter
abstract FrameByFrameConverter <: Converter
abstract TrajectoryConverter <: Converter

include("align.jl")
include("dtw.jl")
include("datasets.jl")
include("feature.jl")
include("gmm.jl")
include("gmmmap.jl")
include("converter.jl")

end # module
