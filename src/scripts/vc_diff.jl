using DocOpt

doc="""Voice conversion based on differencial spectral compensation

Usage:
    vc_diff.jl [options] <input_wav> <model_jld> <dst_wav>
    vc_diff.jl --version
    vc_diff.jl -h | --help

Options:
    -h --help        show this message
    --period=PERIOD  frame period in msec [default: 5.0]
    --order=ORDER    order of mel cepsrum [default: 25]
    --alpha=ALPHA    all-pass constant [default: 0.0]
"""

using VoiceConversion
using MCepAlpha
using WAV
using SPTK
using HDF5, JLD

function main()
    args = docopt(doc, version=v"0.0.1")

    x, fs = wavread(args["<input_wav>"], format="int")
    @assert size(x, 2) == 1 "The input data must be monoral."
    x = float64(x[:])
    const fs = int(fs)
    
    const period = float(args["--period"])
    const order = int(args["--order"])
    alpha = float(args["--alpha"])
    if alpha == 0.0
        alpha = mcepalpha(fs)
    end
    
    # shape (order+1, number of frames)
    src = world_mcep(x, fs, period, order, alpha)
    
    # Load mapping model
    gmm = load(args["<model_jld>"])
    if !gmm["diff"]
        error("not supported")
    end
    mapper = GMMMap(gmm)
    
    # Perform conversion
    converted = vc(mapper, src)

    # remove power coef.
    converted[1,:] = 0.0
    
    # Waveform synthesis using Mel-Log Spectrum Approximation filter
    mf = MLSADF(order)
    hopsize = int(fs / (1000 / period))
    synthesized = synthesis!(mf, x, converted, alpha, hopsize)

    wavwrite(int16(synthesized), args["<dst_wav>"], Fs=fs)
    println("Dumped to ", args["<dst_wav>"])
end

@time main()
