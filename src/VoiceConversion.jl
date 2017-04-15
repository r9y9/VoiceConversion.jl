 __precompile__()

"""
Statistical voice conversion library

[https://github.com/r9y9/VoiceConversion.jl](https://github.com/r9y9/VoiceConversion.jl)

## Design goals

- Modular
- Extendible

## Hierarchy

- Dataset
- Alignment
- Models
- Mapper
- Synthesis

"""
module VoiceConversion

using DocStringExtensions
using StatsBase
using StatsFuns
using Distributions
using MelGeneralizedCepstrums
using HDF5, JLD
using Compat

export FrameByFrameConverter, TrajectoryConverter, GMMMapParam, GMMMap,
    TrajectoryGMMMap, TrajectoryGVGMMMap, fvconvert, vc, ncomponents,
    dim, VarianceScaling, fvpostf!, fvpostf, align, align_mcep,
    Dataset, ParallelDataset, GVDataset, push_delta

for fname in [
              "common",
              "dtw",
              "datasets",
              "align",
              "gmm",
              "gmmmap",
              "diffgmm",
              "trajectory_gmmmap",
              "gv"
    ]
    include(string(fname, ".jl"))
end

end # module
