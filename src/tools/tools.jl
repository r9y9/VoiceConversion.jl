module Tools

# This module provides high-level interfaces for voice conversion

using ..DTWs
using ..VoiceConversion

using MelGeneralizedCepstrums
using WAV
using HDF5, JLD
using WORLD
using PyCall

using Logging
@Logging.configure(level=DEBUG, output=STDOUT)

export
    # Fundamental frequency estimation
    wf0,

    # Spectral envelope estimation
    wmcep,

    # Feature alignment
    align,

    train_gmm,
    save_gmm,

    # utils
    mkdir_if_not_exist

for fname in [
              "util",
              "f0",
              "wmcep",
              "align",
              "train_gmm",
    ]
    include(string(fname, ".jl"))
end

end # module
