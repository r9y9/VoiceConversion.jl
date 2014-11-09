using VoiceConversion
using Base.Test

using MCepAlpha
using WAV
using HDF5, JLD
using SPTK
using WORLD

for fname in ["dtw",
              "vc",
              "diffvc"
              ]
    include(string(fname, ".jl"))
end
