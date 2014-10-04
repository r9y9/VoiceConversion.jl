using VoiceConversion
using Base.Test

for fname in ["dtw",
              "spectrum_differential"]
    include(string(fname, ".jl"))
end
