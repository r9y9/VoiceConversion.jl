using VoiceConversion
using Base.Test

for fname in ["dtw",
              "diffvc"]
    include(string(fname, ".jl"))
end
