using VoiceConversion
using MelGeneralizedCepstrums
using HDF5
using JLD
using SynthesisFilters
using WORLD
using Base.Test

import WORLD: synthesis

for fname in ["dtw",
              "gmmmap",
              "trajectory_gmmmap",
              "vc",
              "diffvc"
              ]
    include(string(fname, ".jl"))
end
