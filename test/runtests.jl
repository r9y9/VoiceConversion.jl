using VoiceConversion
using Base.Test

using MelGeneralizedCepstrums
using HDF5, JLD
using SynthesisFilters

import WORLD: synthesis
using WORLD

for fname in ["dtw",
              "gmmmap",
              "trajectory_gmmmap",
              "vc",
              "diffvc"
              ]
    include(string(fname, ".jl"))
end
