module VoiceConversion

using NumericExtensions
using ArrayViews
using Distributions
using WORLD
using SPTK
using HDF5, JLD

export
    # Voice conversion
    AbstractConverter,
    FrameByFrameConverter,
    TrajectoryConverter,
    GMMMapParam,
    GMMMap,
    TrajectoryGMMMap,
    TrajectoryGVGMMMapp,
    fvconvert,    # feature vector conversion
    vc,           # voice conversion routine
    ncomponents,  # number of mixture components
    dim,

    # Post filters
    PeseudoGV,
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
    mc2e,
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
              "peseudo_gv",
              "vc"]
    include(string(fname, ".jl"))
end

end # module
