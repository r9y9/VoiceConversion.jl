module VoiceConversion

# Voice conversion
export FrameByFrameConverter, TrajectoryConverter,
       fvconvert, vc,
       GMMMap, TrajectoryGMMMap, TrajectoryGMMMapWithGV

# Post filters
export PeseudoGV, fvpostf!, fvpostf

# Dynamic Time Warping (DTW) related functions
export DTW, fit!, update!, set_template!, backward

# Feature conversion, extractions and alignment
export logamp2mc, mc2logamp, mc2e, world_mcep, align_mcep, 
       wsp2mc, mc2wsp

# Datasets
export ParallelDataset, GVDataset, push_delta

## Type Hierarchy ##
abstract Converter
abstract FrameByFrameConverter <: Converter
abstract TrajectoryConverter <: Converter

for fname in ["align",
              "dtw",
              "datasets",
              "mcep",
              "gmm",
              "gmmmap",
              "trajectory_gmmmap",
              "peseudo_gv",
              "converter"]
    include(string(fname, ".jl"))
end

end # module
