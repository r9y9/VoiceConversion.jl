 __precompile__()

module VoiceConversion

using Logging
@Logging.configure(level=DEBUG, output=STDOUT)

using StatsBase
using Distributions
using MelGeneralizedCepstrums
using WORLD
using HDF5, JLD

export
    # Voice conversion
    FrameByFrameConverter,
    TrajectoryConverter,
    GMMMapParam,
    GMMMap,
    TrajectoryGMMMap,
    TrajectoryGVGMMMap,
    fvconvert,    # feature vector conversion
    vc,           # voice conversion routine
    ncomponents,  # number of mixture components
    dim,

    # Post filters
    VarianceScaling,
    fvpostf!,
    fvpostf,

    # Alignment
    align,
    align_mcep,

    # Datasets
    Dataset,
    ParallelDataset,
    GVDataset,
    push_delta

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
