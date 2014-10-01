module VoiceConversion

# Dynamic Time Warping (DTW) related functions
export DTW, fit!, update!, set_template!, backward

# Feature conversion, extractions and alignment
export logamp2mcep, mcep2e, world_mcep, align_mcep

# Datasets
export ParallelDataset, GVDataset, push_delta

# Feature conversion
export Converter, FrameByFrameConverter, TrajectoryConverter,
       GMMMap, GMM, GaussianMixtureModel, fvconvert, vc,
       TrajectoryGMMMap, TrajectoryGMMMapWithGV

# Post filters
export PeseudoGV, fvpostf!, fvpostf

## Type Hierarchy ##
abstract Converter
abstract FrameByFrameConverter <: Converter
abstract TrajectoryConverter <: Converter

for fname in ["align.jl",
              "dtw.jl",
              "datasets.jl",
              "feature.jl",
              "gmm.jl",
              "gmmmap.jl",
              "trajectory_gmmmap.jl",
              "peseudo_gv.jl",
              "converter.jl"]
    include(fname)
end

end # module
