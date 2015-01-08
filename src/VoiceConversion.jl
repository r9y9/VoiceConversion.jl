module VoiceConversion

using NumericExtensions
using ArrayViews
using Distributions
using MelGeneralizedCepstrums
using WORLD
using HDF5, JLD

export
    # Voice conversion
    AbstractConverter,
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
    PeseudoGV,
    VarianceScaling,
    fvpostf!,
    fvpostf,

    # Dynamic Time Warping (DTW) related functions
    DTW,
    fit!,
    update!,
    set_template!,
    backward,

    # Feature conversion, extractions and alignment
    logamp2mc,
    mc2logamp,
    world_mcep,
    align_mcep,
    wsp2mc,
    mc2wsp,

    # Datasets
    ParallelDataset,
    GVDataset,
    push_delta

## Type Hierarchy ##
abstract AbstractConverter
abstract FrameByFrameConverter <: AbstractConverter
abstract TrajectoryConverter <: AbstractConverter

for fname in ["align",
              "dtw",
              "datasets",
              "mcep",
              "gmm",
              "gmmmap",
              "diffgmm",
              "trajectory_gmmmap",
              "gv",
              "vc"]
    include(string(fname, ".jl"))
end

end # module
