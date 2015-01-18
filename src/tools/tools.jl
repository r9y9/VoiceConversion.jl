module Tools

# This module provides high-level interfaces for voice conversion

using ..DTWs
using ..VoiceConversion

using MelGeneralizedCepstrums
using HDF5, JLD
using WORLD
using PyCall

using Logging
@Logging.configure(level=DEBUG, output=STDOUT)

export
    # Fundamental frequency estimation
    wf0,
    save_wf0,

    # Spectral envelope estimation
    wmcep,
    save_wmcep,

    # Feature alignment
    align!,
    save_align,

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
